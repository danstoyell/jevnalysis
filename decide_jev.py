"""Put one question to Jev over a set of options, and print the decision with its
calibrated confidences.

This is the plain version of what the benchmark runners do 62 or 24 times: build
a single `Choice` over the options, send it, and report the full distribution
rather than just the pick. With no options given the question is treated as
yes/no, and a `Noul` -- Jev's "is this statement true?" primitive, which answers
with a bare 0-1 probability and no rubric at all -- rides along as a second,
independent read on the same question. Extra questions are answered in parallel
server-side, so it costs a few tokens and no extra latency.

Options come from a CSV, either a file or inline on the command line:

    a file, one option per line            yes
                                           no

    a file, option,description             ship,"Release to all users today."
                                           hold,"Keep it behind the flag."

    a file, one row of cells               yes,no,maybe

    inline                                 "red,green,blue"

A header row (`option,description` or similar) is detected and skipped. The
description column is worth using: TypeSafe's docs ask you to "describe
situations, not degrees", and bare labels leave the model spreading probability
across options that it cannot tell apart.

--repeats re-asks the same question and averages the per-option probabilities,
then takes the modal option -- the plurality scheme from TypeSafe's
self-consistency cookbook, the same aggregation score_compass.py and score_pew.py
use. It also reports whether the pick flipped between draws, which is the number
to look at before trusting a narrow margin. `Choice` has no notion of ordered
options, but the order they are sent in can still move the answer, so --shuffle
randomises it per draw.

Usage:
    python decide_jev.py "Is the sky blue?"
    python decide_jev.py "Which colour is safest for a warning label?" red,green,blue
    python decide_jev.py "Should we ship?" options.csv --repeats 5 --shuffle
    python decide_jev.py "Is this urgent?" --json --out decision.json
    python decide_jev.py "Is the sky blue?" --dry-run

Docs: https://docs.typesafe.ai/primitives/choice - https://docs.typesafe.ai/primitives/noul
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import random
import sys
import uuid

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from jev_client import JevClient, choice, noul  # noqa: E402

MAX_OPTIONS = 255  # Choice's published ceiling

INSTRUCTIONS = "Select the option that best answers the question given in the state."
NOUL_INSTRUCTIONS = "The answer to the question given in the state is yes."

YES_NO: dict[str, str | None] = {
    "yes": "The answer to the question given in the state is yes.",
    "no": "The answer to the question given in the state is no.",
}

HEADER_LABELS = {"option", "options", "label", "labels", "choice", "choices", "answer", "criteria"}
HEADER_DESCS = {"description", "descriptions", "desc", "detail", "details", "meaning", "criterion"}


def load_dotenv(path: pathlib.Path) -> None:
    """Fill in unset vars from a .env file. The benchmark runners expect you to
    export the key yourself; this is a one-off CLI, so be friendlier about it."""
    import os

    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def parse_options(spec: str | None) -> tuple[dict[str, str | None], str]:
    """Turn the CSV argument into Choice criteria. Returns (criteria, source)."""
    if spec is None:
        return dict(YES_NO), "default yes/no"

    path = pathlib.Path(spec)
    is_file = path.exists()
    if is_file:
        text = path.read_text()
        source = f"{path}"
    else:
        # Don't silently treat a mistyped path as a one-item inline list.
        if "\n" in spec or "/" in spec or spec.lower().endswith((".csv", ".tsv", ".txt")):
            raise SystemExit(f"no such file: {spec!r}")
        text = spec
        source = "inline"

    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows:
        raise SystemExit(f"no options found in {source}")

    # option,description per row -- but a single row of bare cells is a list of
    # options, not one option with 3 descriptions.
    two_col = len(rows) > 1 and max(len(r) for r in rows) >= 2

    # Only a file can have a header row; an inline list is all options, so
    # stripping one there would silently eat an option called "choice".
    header = is_file and len(rows) > 1

    if two_col:
        if header and len(rows[0]) >= 2 and rows[0][0].strip().lower() in HEADER_LABELS \
                and rows[0][1].strip().lower() in HEADER_DESCS:
            rows = rows[1:]
        pairs = [(r[0].strip(), (r[1].strip() if len(r) > 1 and r[1].strip() else None)) for r in rows]
    else:
        if header and len(rows[0]) == 1 and rows[0][0].strip().lower() in HEADER_LABELS:
            rows = rows[1:]
        cells = [c.strip() for r in rows for c in r if c.strip()]
        pairs = [(c, None) for c in cells]

    pairs = [(lab, d) for lab, d in pairs if lab]
    labels = [lab for lab, _ in pairs]
    dupes = sorted({lab for lab in labels if labels.count(lab) > 1})
    if dupes:
        raise SystemExit(f"duplicate options in {source}: {', '.join(dupes)}")
    if len(pairs) < 2:
        raise SystemExit(f"need at least 2 options, got {len(pairs)} from {source}")
    if len(pairs) > MAX_OPTIONS:
        raise SystemExit(f"Choice takes at most {MAX_OPTIONS} options, got {len(pairs)}")
    return dict(pairs), source


def aggregate(draws: list[dict]) -> dict:
    """Average per-option probabilities across draws, then take the modal option.
    Matches the aggregation in score_compass.py / score_pew.py."""
    probs: dict[str, float] = {}
    for d in draws:
        for label, p in d["probabilities"].items():
            probs[label] = probs.get(label, 0.0) + p / len(draws)
    modal = max(probs, key=lambda k: probs[k])
    out = {
        "decision": modal,
        "probabilities": probs,
        "decision_prob": probs[modal],
        "mean_confidence": sum(d["confidence"] for d in draws) / len(draws),
        "n_draws": len(draws),
        "flipped": len({d["choice"] for d in draws}) > 1,
        "picks": [d["choice"] for d in draws],
    }
    nouls = [d["noul"] for d in draws if d.get("noul") is not None]
    if nouls:
        out["mean_noul"] = sum(nouls) / len(nouls)
    return out


def report(question: str, agg: dict, criteria: dict[str, str | None], source: str) -> str:
    lines = [f"Q: {question}", f"   {len(criteria)} options ({source})", ""]
    ranked = sorted(agg["probabilities"].items(), key=lambda kv: -kv[1])
    width = max(len(k) for k in agg["probabilities"])
    for label, p in ranked:
        bar = "#" * round(p * 30)
        mark = "->" if label == agg["decision"] else "  "
        lines.append(f" {mark} {label:<{width}}  {p:5.2f}  {bar}")
    lines += ["", f"Decision: {agg['decision']}  (p={agg['decision_prob']:.2f}, "
                  f"confidence={agg['mean_confidence']:.2f})"]
    if "mean_noul" in agg:
        lines.append(f"Noul cross-check: P(answer is yes) = {agg['mean_noul']:.2f}")
    if agg["n_draws"] > 1:
        flip = "FLIPPED between draws" if agg["flipped"] else "stable across draws"
        lines.append(f"{agg['n_draws']} draws, {flip}: {', '.join(agg['picks'])}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("question", help="the question to put to Jev")
    ap.add_argument("options", nargs="?",
                    help="CSV file of options, or an inline comma-separated list; "
                         "omit for yes/no")
    ap.add_argument("--provider", default="openrouter", choices=["openrouter", "typesafe"])
    ap.add_argument("--repeats", type=int, default=1, help="re-ask and average the distributions")
    ap.add_argument("--shuffle", action="store_true", help="randomise option order per draw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-noul", action="store_true",
                    help="skip the yes/no Noul cross-check")
    ap.add_argument("--json", action="store_true", help="print JSON instead of the table")
    ap.add_argument("--out", help="also write the full result, including every draw, here")
    ap.add_argument("--dry-run", action="store_true", help="print the payload, send nothing")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    criteria, source = parse_options(args.options)
    # The Noul is only meaningful when the question really is yes/no.
    use_noul = args.options is None and not args.no_noul
    rng = random.Random(args.seed)

    client = JevClient(provider=args.provider, dry_run=args.dry_run)
    draws: list[dict] = []

    for rep in range(args.repeats):
        items = list(criteria.items())
        if args.shuffle:
            rng.shuffle(items)
        questions: dict[str, dict] = {"answer": choice(INSTRUCTIONS, dict(items))}
        if use_noul:
            questions["is_yes"] = noul(NOUL_INSTRUCTIONS)

        state: dict = {"question": args.question}
        if args.repeats > 1:
            # Distinct state per draw, as the runners do, so repeats are real draws.
            state["uid"] = str(uuid.uuid4())

        resp = client.ask(state, questions)
        if args.dry_run:
            print(json.dumps(resp["request"], indent=2, ensure_ascii=False))
            print(f"\nDry run: 1 of {args.repeats} request(s) built, 0 sent.", file=sys.stderr)
            return

        a = resp["answers"]["answer"]
        draws.append({
            "repeat": rep,
            "choice": a["choice"],
            "probabilities": a["probabilities"],
            "confidence": a["confidence"],
            "noul": resp["answers"]["is_yes"]["noul"] if use_noul else None,
            "option_order": [k for k, _ in items],
        })

    agg = aggregate(draws)
    result = {
        "question": args.question,
        "options": criteria,
        "options_source": source,
        "provider": args.provider,
        "model": client.model,
        "repeats": args.repeats,
        "shuffled": args.shuffle,
        **agg,
        "usage": client.usage,
        "cost_usd": round(client.cost_usd(), 6),
        "draws": draws,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(report(args.question, agg, criteria, source), flush=True)
        print(f"\n{client.calls} call(s), ${client.cost_usd():.4f}.", file=sys.stderr)

    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
