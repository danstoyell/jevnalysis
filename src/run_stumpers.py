"""Put 20 altered classic puzzles to Jev and see whether it picks the lure.

Each item is a well-known riddle or thought experiment with one detail changed.
The option set always contains the correct answer to the altered question and
the LURE -- the memorised answer to the unaltered original. Picking the lure is
the diagnostic failure: pattern recognition beating reading.

TypeSafe markets Jev as a "System One" model, and these items were built by
researchers to catch exactly System 1 pattern-matching, so this runs close to an
adversarial test of the product's own framing.

Option order is shuffled per draw, since `Choice` has no notion of ordering and
position should carry no signal.

Usage:
    python src/run_stumpers.py --repeats 5
    python src/run_stumpers.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from jev_client import JevClient, choice  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
INSTRUCTIONS = "Answer the question given in the state. Select the option that is correct."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="openrouter", choices=["openrouter", "typesafe"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "stumpers_raw.json"))
    args = ap.parse_args()

    bank = json.loads((ROOT / "data" / "stumpers.json").read_text())
    items = bank["items"][: args.limit] if args.limit else bank["items"]
    rng = random.Random(args.seed)

    client = JevClient(provider=args.provider, dry_run=args.dry_run)
    records = []

    for i, item in enumerate(items, 1):
        for rep in range(args.repeats):
            opts = list(item["options"])
            rng.shuffle(opts)
            state = {"question": item["question"]}
            if args.repeats > 1:
                state["uid"] = str(uuid.uuid4())

            resp = client.ask(state, {"answer": choice(INSTRUCTIONS, {o: None for o in opts})})

            if args.dry_run:
                if i == 1 and rep == 0:
                    print(json.dumps(resp["request"], indent=2, ensure_ascii=False))
                records.append({"id": item["id"], "request": resp["request"]})
                continue

            a = resp["answers"]["answer"]
            picked = a["choice"]
            records.append({
                "id": item["id"],
                "source": item["source"],
                "question": item["question"],
                "repeat": rep,
                "picked": picked,
                "correct_answer": item["correct"],
                "lure_answer": item["lure"],
                "is_correct": picked == item["correct"],
                "is_lure": picked == item["lure"],
                "p_correct": a["probabilities"].get(item["correct"], 0.0),
                "p_lure": a["probabilities"].get(item["lure"], 0.0),
                "probabilities": a["probabilities"],
                "confidence": a["confidence"],
                "option_order": opts,
            })
            if rep == args.repeats - 1:
                tag = "CORRECT" if picked == item["correct"] else ("LURE" if picked == item["lure"] else "other")
                print(f"[{i:2d}/{len(items)}] {item['id']:<26} {tag:<8} "
                      f"p_correct={a['probabilities'].get(item['correct'],0):.2f} "
                      f"p_lure={a['probabilities'].get(item['lure'],0):.2f}", flush=True)

    out = {
        "benchmark": "llm_stumpers",
        "provider": args.provider,
        "model": client.model,
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
