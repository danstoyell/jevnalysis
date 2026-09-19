"""Compass coordinates from the no-refusal control condition, for any model.

The headline comparison has to hold the method fixed. Jev's original coordinates
came from the `Score` primitive; the chat models cannot use `Score` at all. The
control condition -- four Likert options, no refusal, presented as an unordered
choice -- is the one condition every model ran identically, so it is what the
cross-model compass table is built from.

Scoring is still the live politicalcompass.org form walk, as in score_compass.py.

Usage:
    python src/score_compass_control.py --in results/compass_refusal_raw.json --label "Jev 1.13"
    python src/score_compass_control.py --all
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from score_compass import submit  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
LIKERT = ["Strongly disagree", "Disagree", "Agree", "Strongly agree"]


def control_answers(raw: dict) -> tuple[dict[str, int], dict]:
    """Modal Likert answer per item in the control condition, as form values 0-3."""
    per_item: dict[str, list[dict]] = defaultdict(list)
    for r in raw["records"]:
        if r.get("condition") == "control":
            per_item[r["id"]].append(r)

    answers, detail = {}, {}
    for item_id, draws in per_item.items():
        probs: dict[str, float] = defaultdict(float)
        for d in draws:
            for k, v in d["probabilities"].items():
                probs[k] += v / len(draws)
        modal = max(probs, key=lambda k: probs[k])
        if modal not in LIKERT:
            raise SystemExit(f"{item_id}: control produced a non-Likert answer {modal!r}")
        answers[item_id] = LIKERT.index(modal)
        detail[item_id] = {
            "answer": LIKERT.index(modal),
            "label": modal,
            "probabilities": dict(probs),
            "confidence": sum(d["confidence"] for d in draws) / len(draws),
        }
    return answers, detail


def score_one(path: str, label: str | None = None) -> dict:
    raw = json.loads(pathlib.Path(path).read_text())
    name = label or raw["model"]
    answers, detail = control_answers(raw)
    if len(answers) != 62:
        raise SystemExit(f"{name}: expected 62 control items, got {len(answers)}")

    bank = json.loads((ROOT / "data" / "political_compass.json").read_text())
    print(f"\n{name} -- submitting 62 control answers")
    coords = submit(answers, bank)
    print(f"  Economic {coords['economic']:+.2f}   Social {coords['social']:+.2f}")

    mean_conf = sum(v["confidence"] for v in detail.values()) / len(detail)
    return {
        "model": raw["model"],
        "label": name,
        "repeats": raw["repeats"],
        "condition": "control",
        "economic": coords["economic"],
        "social": coords["social"],
        "mean_confidence": round(mean_conf, 4),
        "items": detail,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="infile")
    ap.add_argument("--label")
    ap.add_argument("--all", action="store_true", help="score every compass_refusal_*_raw.json present")
    ap.add_argument("--out", default=str(ROOT / "results" / "compass_control_all.json"))
    args = ap.parse_args()

    targets: list[tuple[str, str | None]] = []
    if args.all:
        jev = ROOT / "results" / "compass_refusal_raw.json"
        if jev.exists():
            targets.append((str(jev), "Jev 1.13"))
        for p in sorted(glob.glob(str(ROOT / "results" / "compass_refusal_*_raw.json"))):
            if pathlib.Path(p).name != "compass_refusal_raw.json":
                targets.append((p, None))
    elif args.infile:
        targets.append((args.infile, args.label))
    else:
        raise SystemExit("pass --in or --all")

    out = {}
    for path, label in targets:
        try:
            r = score_one(path, label)
            out[r["label"]] = r
        except SystemExit as e:
            print(f"  skipped {pathlib.Path(path).name}: {e}")

    if out:
        print(f"\n{'model':<26}{'economic':>10}{'social':>9}{'mean conf':>11}")
        for k, v in out.items():
            print(f"{k:<26}{v['economic']:>+10.2f}{v['social']:>+9.2f}{v['mean_confidence']:>11.2f}")
        prev = pathlib.Path(args.out)
        merged = json.loads(prev.read_text()) if prev.exists() else {}
        merged.update(out)
        prev.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
