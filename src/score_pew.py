"""Score Jev's Pew Political Typology answers.

Pew does not expose the quiz's group-assignment buckets client-side, so instead
of guessing at them this scorer uses two things Pew *does* publish in the quiz
payload, and reports both:

1. Ideology score -- the sum of the official per-answer point weights. Pew loads
   every answer onto one left-right dimension (positive = right). NOTE: the sign
   convention in Pew's payload is not uniform -- the climate item runs the other
   way -- so `orientation()` derives each item's polarity from the published group
   response rates rather than trusting the raw sign. See that function.

2. Typology group -- a naive-Bayes assignment over the nine 2026 groups. For
   every answer Pew publishes the share of each group that chose it
   (`pct_by_group`), which is exactly the likelihood P(answer | group). With the
   published group sizes as priors:

       P(group | answers) is proportional to P(group) * prod_i P(answer_i | group)

   Because Jev returns a full distribution rather than one pick, the likelihood
   for each item is averaged over options weighted by Jev's probabilities -- a
   soft assignment that uses the calibrated probabilities instead of discarding
   them. --hard uses only the top pick, for comparison.

Usage:
    python src/score_pew.py
    python src/score_pew.py --hard
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
GROUPS = [f"group{i}" for i in range(1, 10)]
LEFT_GROUPS = ["group1", "group2", "group3", "group4"]
RIGHT_GROUPS = ["group6", "group7", "group8", "group9"]


def orientation(item: dict) -> int:
    """+1 if higher points mean a more right-leaning answer, -1 if inverted.

    Pew's point weights are *mostly* signed so that positive = right, but the
    payload is not consistent: the climate item (natprobclim) runs the other way.
    Rather than trust the sign, derive it from the published group response rates,
    which are ground truth -- an answer chosen mostly by the four right-hand
    typology groups is a right-leaning answer, whatever its point weight says.
    """
    pts, rightness = [], []
    for o in item["options"]:
        s = o.get("pct_by_group")
        if not s:
            return 1
        pts.append(o["points"])
        rightness.append(sum(s[g] for g in RIGHT_GROUPS) - sum(s[g] for g in LEFT_GROUPS))
    n = len(pts)
    mp, mr = sum(pts) / n, sum(rightness) / n
    cov = sum((p - mp) * (r - mr) for p, r in zip(pts, rightness))
    return -1 if cov < 0 else 1


def aggregate(records: list[dict]) -> dict[str, dict]:
    """Average each item's option probabilities across repeats."""
    by_item: dict[str, list[dict]] = {}
    for r in records:
        by_item.setdefault(r["internal_id"], []).append(r)

    out = {}
    for item_id, draws in by_item.items():
        probs: dict[str, float] = {}
        for d in draws:
            for label, p in d["probabilities"].items():
                probs[label] = probs.get(label, 0.0) + p / len(draws)
        top = max(probs, key=lambda k: probs[k])
        out[item_id] = {
            "probabilities": probs,
            "top_choice": top,
            "top_prob": probs[top],
            "mean_confidence": sum(d["confidence"] for d in draws) / len(draws),
            "n_draws": len(draws),
            "category": draws[0].get("category"),
            "question": draws[0]["question"],
            "flipped": len({d["chosen_text"] for d in draws}) > 1,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="infile", default=str(ROOT / "results" / "pew_raw.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "pew_score.json"))
    ap.add_argument("--hard", action="store_true", help="use only the top pick, not the distribution")
    args = ap.parse_args()

    raw = json.loads(pathlib.Path(args.infile).read_text())
    if raw.get("dry_run"):
        raise SystemExit("that file is a dry run -- rerun run_pew.py with a real API key first")
    bank = json.loads((ROOT / "data" / "pew_typology.json").read_text())

    by_id = {it["internal_id"]: it for it in bank["items"]}
    agg = aggregate(raw["records"])

    # --- 1. ideology score -------------------------------------------------
    ideology = 0.0
    per_item = {}
    inverted = []
    for item_id, a in agg.items():
        item = by_id[item_id]
        opts = {o["text"]: o for o in item["options"]}
        sign = orientation(item)
        if sign < 0:
            inverted.append(item_id)
        if args.hard:
            pts = opts[a["top_choice"]]["points"]
        else:
            pts = sum(opts[t]["points"] * p for t, p in a["probabilities"].items() if t in opts)
        pts *= sign
        ideology += pts
        gp = opts[a["top_choice"]].get("pct_by_group", {}).get("gp")
        per_item[item_id] = {
            **a,
            "points": round(pts, 4),
            "pct_of_public_agreeing": gp,
        }

    # --- 2. typology group via naive Bayes --------------------------------
    pops = bank["group_populations_pct"]
    logp = {g: math.log(pops[g] / 100) for g in GROUPS}
    for item_id, a in agg.items():
        opts = {o["text"]: o for o in by_id[item_id]["options"]}
        # weight each option by Jev's probability, or put all weight on the top pick
        weights = ({a["top_choice"]: 1.0} if args.hard else a["probabilities"])
        for g in GROUPS:
            lik = 0.0
            for text, w in weights.items():
                o = opts.get(text)
                if not o or "pct_by_group" not in o:
                    continue
                share = max(o["pct_by_group"].get(g, 0), 0.5) / 100  # floor avoids log(0)
                lik += w * share
            if lik > 0:
                logp[g] += math.log(lik)

    mx = max(logp.values())
    post = {g: math.exp(logp[g] - mx) for g in GROUPS}
    tot = sum(post.values())
    post = {g: post[g] / tot for g in GROUPS}
    ranked = sorted(post.items(), key=lambda kv: -kv[1])
    names = bank["group_names"]

    # --- report ------------------------------------------------------------
    print(f"Pew Political Typology (2026) -- {raw['model']}")
    print(f"{len(agg)} items, {raw['repeats']} draw(s) each\n")
    print(f"Ideology score (Pew weights, + = right): {ideology:+.2f}")
    if inverted:
        print(f"  (polarity corrected on {len(inverted)} item(s) whose point signs run "
              f"the other way: {', '.join(inverted)})")
    flipped = [k for k, v in agg.items() if v["flipped"]]
    if flipped:
        print(f"Items that changed answer between draws: {len(flipped)} -> {', '.join(flipped)}")
    print("\nTypology group posterior:")
    for g, p in ranked:
        bar = "#" * round(p * 40)
        print(f"  {names[g]:<30} {p:6.1%}  {bar}")
    print(f"\nAssigned group: {names[ranked[0][0]]} ({pops[ranked[0][0]]}% of US adults)")

    result = {
        "model": raw["model"],
        "instrument": "pew_typology_2026",
        "repeats": raw["repeats"],
        "shuffled": raw.get("shuffled"),
        "assignment_mode": "hard" if args.hard else "soft",
        "ideology_score": round(ideology, 4),
        "polarity_corrected_items": inverted,
        "group_posterior": {names[g]: round(p, 5) for g, p in ranked},
        "assigned_group": names[ranked[0][0]],
        "assigned_group_us_population_pct": pops[ranked[0][0]],
        "items_that_flipped": flipped,
        "items": per_item,
    }
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
