"""Administer the Pew Research Center Political Typology Quiz (2026) to Jev.

Every Pew item is a closed multiple choice, so each maps onto a single Jev
Choice question with the answer options as criteria. As with the compass, one
request per item keeps items independent.

Pew's option order is meaningful (it runs from one pole to the other), and
Choice has no notion of order, so option order is an experimental nuisance
here. --shuffle randomises it per draw; combined with --repeats it measures how
much of the result is order artefact.

Usage:
    python src/run_pew.py --dry-run
    python src/run_pew.py --repeats 5 --shuffle
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

INSTRUCTIONS = "Answer the survey question given in the state. Select the option that comes closest."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="openrouter", choices=["openrouter", "typesafe"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--shuffle", action="store_true", help="randomise option order per draw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(ROOT / "results" / "pew_raw.json"))
    args = ap.parse_args()

    bank = json.loads((ROOT / "data" / "pew_typology.json").read_text())
    items = bank["items"][: args.limit] if args.limit else bank["items"]
    rng = random.Random(args.seed)

    client = JevClient(provider=args.provider, dry_run=args.dry_run)
    records = []

    for i, item in enumerate(items, 1):
        for rep in range(args.repeats):
            opts = list(item["options"])
            if args.shuffle:
                rng.shuffle(opts)

            # Criteria keys are the option uuids so we can map an answer back to
            # its official Pew point weight without depending on the label text.
            criteria = {o["uuid"]: o["text"] for o in opts}

            question_text = item["text"]
            if item.get("preamble"):
                question_text = f"{item['preamble']} {question_text}"

            state = {"question": question_text}
            if args.repeats > 1:
                state["uid"] = str(uuid.uuid4())

            resp = client.ask(state, {"answer": choice(INSTRUCTIONS, criteria)})

            if args.dry_run:
                if i == 1 and rep == 0:
                    print("Example request payload:\n")
                    print(json.dumps(resp["request"], indent=2, ensure_ascii=False))
                    print("\n" + "-" * 60)
                records.append({"internal_id": item["internal_id"], "repeat": rep, "request": resp["request"]})
                continue

            ans = resp["answers"]["answer"]
            picked = {o["uuid"]: o for o in item["options"]}[ans["choice"]]
            records.append({
                "internal_id": item["internal_id"],
                "category": item.get("category"),
                "question": question_text,
                "repeat": rep,
                "chosen_uuid": ans["choice"],
                "chosen_text": picked["text"],
                "points": picked["points"],
                "confidence": ans["confidence"],
                # remap probabilities from uuid -> readable label
                "probabilities": {
                    {o["uuid"]: o["text"] for o in item["options"]}[u]: p
                    for u, p in ans["probabilities"].items()
                },
                "option_order": [o["uuid"] for o in opts],
            })
            print(f"[{i:2d}/{len(items)}] {str(item['internal_id']):<20} "
                  f"pts={picked['points']:+.2f} conf={ans['confidence']:.2f}  {picked['text'][:52]}")

    out = {
        "instrument": "pew_typology_2026",
        "provider": args.provider,
        "model": client.model,
        "repeats": args.repeats,
        "shuffled": args.shuffle,
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
        print(f"\n{client.calls} calls, ${client.cost_usd():.4f}. Wrote {args.out}")
        print("Next: python src/score_pew.py")


if __name__ == "__main__":
    main()
