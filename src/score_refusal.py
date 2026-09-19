"""Summarise how often Jev takes a refusal option, by phrasing.

Two different questions, both reported, because they can disagree:

  pick rate   share of the 62 items whose MODAL answer is the refusal option
              -- "how often would a respondent have refused"
  mass        mean probability Jev puts on the refusal option across all items
              -- "how much pull the option exerts", which the pick rate hides
              when refusal is consistently second place

The control condition has no refusal option; its Likert distribution is the
baseline the other three shift away from.

Usage:
    python src/score_refusal.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="infile", default=str(ROOT / "results" / "compass_refusal_raw.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "compass_refusal_score.json"))
    args = ap.parse_args()

    raw = json.loads(pathlib.Path(args.infile).read_text())
    likert = raw["likert"]

    # average each item's option probabilities across draws, within condition
    agg: dict[str, dict[str, dict]] = defaultdict(dict)
    per_item_draws: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in raw["records"]:
        per_item_draws[(r["condition"], r["id"])].append(r)

    for (cond, item_id), draws in per_item_draws.items():
        probs: dict[str, float] = defaultdict(float)
        for d in draws:
            for k, v in d["probabilities"].items():
                probs[k] += v / len(draws)
        modal = max(probs, key=lambda k: probs[k])
        agg[cond][item_id] = {
            "probabilities": dict(probs),
            "modal": modal,
            "refusal_prob": sum(v for k, v in probs.items() if k not in likert),
            "text": draws[0]["text"],
            "confidence": sum(d["confidence"] for d in draws) / len(draws),
        }

    summary = {}
    for cond, items in agg.items():
        label = raw["conditions"][cond]
        n = len(items)
        picked = [k for k, v in items.items() if v["modal"] not in likert]
        mass = sum(v["refusal_prob"] for v in items.values()) / n
        summary[cond] = {
            "refusal_label": label,
            "n_items": n,
            "pick_count": len(picked),
            "pick_rate": len(picked) / n,
            "mean_refusal_mass": mass,
            "max_refusal_mass": max(v["refusal_prob"] for v in items.values()),
            "mean_confidence": sum(v["confidence"] for v in items.values()) / n,
            "items_picked": picked,
            "top_refusal_items": sorted(
                ({"id": k, "p": round(v["refusal_prob"], 3), "text": v["text"]} for k, v in items.items()),
                key=lambda x: -x["p"],
            )[:5],
        }

    print(f"Political Compass with a refusal option -- {raw['model']}")
    print(f"{raw['n_items']} items, {raw['repeats']} draws, Choice primitive\n")
    print(f"{'condition':<12}{'label':<22}{'picked':>8}{'pick rate':>11}{'mean mass':>11}{'max mass':>10}{'conf':>7}")
    for cond in ["control", "prefer", "undecided", "noopinion"]:
        if cond not in summary:
            continue
        s = summary[cond]
        print(f"{cond:<12}{str(s['refusal_label'] or '(none)'):<22}"
              f"{s['pick_count']:>8}{s['pick_rate']:>10.1%}{s['mean_refusal_mass']:>11.3f}"
              f"{s['max_refusal_mass']:>10.3f}{s['mean_confidence']:>7.2f}")

    for cond in ["prefer", "undecided", "noopinion"]:
        if cond not in summary:
            continue
        print(f"\n{summary[cond]['refusal_label']} -- highest refusal mass:")
        for t in summary[cond]["top_refusal_items"]:
            print(f"  {t['p']:.2f}  {t['text'][:76]}")

    pathlib.Path(args.out).write_text(json.dumps(
        {"model": raw["model"], "repeats": raw["repeats"], "summary": summary, "items": agg},
        indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
