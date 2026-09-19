"""Administer the 62-item Political Compass test to Jev.

Each proposition is sent as its own request so items cannot contaminate each
other (the docs warn that "accuracy falls as the state grows with content
unrelated to the decision"). Two questions ride along in every call, since
extra questions are answered in parallel and barely change latency:

  agreement  Score over the test's own 4-point scale -> used for official scoring
  agrees     Noul, a continuous 0-1 agreement probability -> robustness check

Usage:
    python src/run_compass.py --dry-run             # inspect payloads, no key needed
    python src/run_compass.py --repeats 5           # 5 independent draws per item
    python src/run_compass.py --rubric anchored     # expanded level descriptions
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from jev_client import JevClient, noul, score  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent

# The instrument's own labels, verbatim. Faithful to how a human sees the test.
RUBRIC_PLAIN = ["Strongly disagree", "Disagree", "Agree", "Strongly agree"]

# Jev's docs say to "describe situations, not degrees", and warn that bare
# descriptors leave the model spreading probability across levels. This variant
# keeps the same four levels but anchors them, as a sensitivity check.
RUBRIC_ANCHORED = [
    "Strongly disagree: the proposition is plainly wrong and should be rejected outright",
    "Disagree: the proposition is more wrong than right, though not absurd",
    "Agree: the proposition is more right than wrong, though not beyond question",
    "Strongly agree: the proposition is plainly right and should be endorsed outright",
]

RUBRICS = {"plain": RUBRIC_PLAIN, "anchored": RUBRIC_ANCHORED}

INSTRUCTIONS = "Indicate the level of agreement with the proposition given in the state."
NOUL_INSTRUCTIONS = "The proposition given in the state is one to agree with."


def build_questions(rubric: str) -> dict:
    return {
        "agreement": score(INSTRUCTIONS, RUBRICS[rubric]),
        "agrees": noul(NOUL_INSTRUCTIONS),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="openrouter", choices=["openrouter", "typesafe"])
    ap.add_argument("--rubric", default="plain", choices=list(RUBRICS))
    ap.add_argument("--repeats", type=int, default=1, help="independent draws per item (self-consistency)")
    ap.add_argument("--dry-run", action="store_true", help="print payloads, make no API calls")
    ap.add_argument("--limit", type=int, help="only run the first N items (smoke test)")
    ap.add_argument("--out", default=str(ROOT / "results" / "compass_raw.json"))
    args = ap.parse_args()

    bank = json.loads((ROOT / "data" / "political_compass.json").read_text())
    items = bank["items"][: args.limit] if args.limit else bank["items"]
    questions = build_questions(args.rubric)

    client = JevClient(provider=args.provider, dry_run=args.dry_run)
    records = []

    for i, item in enumerate(items, 1):
        for rep in range(args.repeats):
            # A throwaway uid makes each repeat an independent draw rather than a
            # cache replay -- the technique from TypeSafe's self-consistency cookbook.
            state = {"proposition": item["text"]}
            if args.repeats > 1:
                state["uid"] = str(uuid.uuid4())

            resp = client.ask(state, questions)

            if args.dry_run:
                if i == 1 and rep == 0:
                    print("Example request payload:\n")
                    print(json.dumps(resp["request"], indent=2, ensure_ascii=False))
                    print("\n" + "-" * 60)
                records.append({"id": item["id"], "repeat": rep, "request": resp["request"]})
                continue

            ans = resp["answers"]
            records.append({
                "id": item["id"],
                "page": item["page"],
                "text": item["text"],
                "repeat": rep,
                "score": ans["agreement"]["score"],
                "score_probabilities": ans["agreement"]["probabilities"],
                "score_confidence": ans["agreement"]["confidence"],
                "noul": ans["agrees"]["noul"],
            })
            print(f"[{i:2d}/{len(items)}] {item['id']:<26} "
                  f"score={ans['agreement']['score']:.2f} "
                  f"conf={ans['agreement']['confidence']:.2f} "
                  f"noul={ans['agrees']['noul']:.2f}")

    out = {
        "instrument": "political_compass",
        "provider": args.provider,
        "model": client.model,
        "rubric": args.rubric,
        "rubric_levels": RUBRICS[args.rubric],
        "repeats": args.repeats,
        "n_items": len(items),
        "dry_run": args.dry_run,
        "usage": client.usage,
        "cost_usd": round(client.cost_usd(), 6),
        "records": records,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    if args.dry_run:
        print(f"\nDry run: {len(records)} requests built, 0 sent. Payloads -> {args.out}")
    else:
        print(f"\n{client.calls} calls, {client.usage['input_tokens']:,} input tokens, "
              f"${client.cost_usd():.4f}. Wrote {args.out}")
        print("Next: python src/score_compass.py")


if __name__ == "__main__":
    main()
