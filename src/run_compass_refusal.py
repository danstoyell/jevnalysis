"""Re-run the Political Compass with an explicit refusal option, three phrasings.

A refusal is not a point on the agree/disagree spectrum, so `Score` is the wrong
primitive here -- an ordered rubric would force "Prefer not to answer" onto the
scale between two Likert levels. Each proposition therefore becomes a `Choice`
over unordered options: the instrument's own four labels, plus (in three of the
four conditions) one refusal option.

Conditions:
    control   4 options, no refusal -- the baseline that makes the rest readable
    prefer    + "Prefer not to answer"
    undecided + "Undecided"
    noopinion + "No Opinion"

Criteria values are null so the model sees the bare labels, matching the `plain`
rubric used in the Score runs.

Usage:
    python src/run_compass_refusal.py --repeats 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from jev_client import JevClient, choice  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent

LIKERT = ["Strongly disagree", "Disagree", "Agree", "Strongly agree"]

CONDITIONS = {
    "control": None,
    "prefer": "Prefer not to answer",
    "undecided": "Undecided",
    "noopinion": "No Opinion",
}

INSTRUCTIONS = "Select the response that best matches the level of agreement with the proposition given in the state."


def build(refusal: str | None) -> dict:
    opts = {lab: None for lab in LIKERT}
    if refusal:
        opts[refusal] = None
    return {"response": choice(INSTRUCTIONS, opts)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="openrouter", choices=["openrouter", "typesafe"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "compass_refusal_raw.json"))
    args = ap.parse_args()

    bank = json.loads((ROOT / "data" / "political_compass.json").read_text())
    items = bank["items"][: args.limit] if args.limit else bank["items"]
    client = JevClient(provider=args.provider, dry_run=args.dry_run)

    records = []
    for cond, refusal in CONDITIONS.items():
        questions = build(refusal)
        for i, item in enumerate(items, 1):
            for rep in range(args.repeats):
                state = {"proposition": item["text"]}
                if args.repeats > 1:
                    state["uid"] = str(uuid.uuid4())
                resp = client.ask(state, questions)
                if args.dry_run:
                    records.append({"condition": cond, "id": item["id"], "request": resp["request"]})
                    continue
                a = resp["answers"]["response"]
                records.append({
                    "condition": cond,
                    "refusal_label": refusal,
                    "id": item["id"],
                    "page": item["page"],
                    "text": item["text"],
                    "repeat": rep,
                    "choice": a["choice"],
                    "probabilities": a["probabilities"],
                    "confidence": a["confidence"],
                    "refusal_prob": a["probabilities"].get(refusal, 0.0) if refusal else 0.0,
                })
            if not args.dry_run and i % 20 == 0:
                print(f"  {cond}: {i}/{len(items)}", flush=True)
        print(f"{cond} done ({len(items)} items x {args.repeats} draws)", flush=True)

    out = {
        "instrument": "political_compass_refusal",
        "provider": args.provider,
        "model": client.model,
        "primitive": "choice",
        "likert": LIKERT,
        "conditions": CONDITIONS,
        "repeats": args.repeats,
        "n_items": len(items),
        "dry_run": args.dry_run,
        "usage": client.usage,
        "cost_usd": round(client.cost_usd(), 6),
        "records": records,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n{client.calls} calls, ${client.cost_usd():.4f}. Wrote {args.out}")


if __name__ == "__main__":
    main()
