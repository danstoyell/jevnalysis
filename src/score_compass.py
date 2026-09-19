"""Score Jev's Political Compass answers using the test's own scoring engine.

The official axis weights have never been published, so any local reimplementation
would be a guess. The live test is a plain 6-page HTML form that carries running
totals in hidden `carried_ec` / `carried_soc` fields, so we can walk the pages
over HTTP and let politicalcompass.org do the arithmetic. No browser needed.

Aggregation across repeats: each item's per-level probabilities are averaged over
draws, then the modal level is taken as that item's answer. This is the
plurality-vote scheme from TypeSafe's self-consistency cookbook.

Usage:
    python src/score_compass.py
    python src/score_compass.py --in results/compass_raw.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).parent.parent
TEST_URL = "https://www.politicalcompass.org/test/en"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


def aggregate(records: list[dict]) -> dict[str, dict]:
    """Collapse repeats into one 0-3 answer per item."""
    by_item: dict[str, list[dict]] = {}
    for r in records:
        by_item.setdefault(r["id"], []).append(r)

    out = {}
    for item_id, draws in by_item.items():
        mean_probs: dict[str, float] = {}
        for d in draws:
            for level, p in d["score_probabilities"].items():
                mean_probs[level] = mean_probs.get(level, 0.0) + p / len(draws)
        modal = max(mean_probs, key=lambda k: mean_probs[k])
        out[item_id] = {
            "answer": int(modal),
            "mean_probabilities": mean_probs,
            "mean_score": sum(d["score"] for d in draws) / len(draws),
            "mean_noul": sum(d["noul"] for d in draws) / len(draws),
            "mean_confidence": sum(d["score_confidence"] for d in draws) / len(draws),
            "n_draws": len(draws),
            "modal_share": mean_probs[modal],
        }
    return out


def post(data: dict[str, str]) -> str:
    req = urllib.request.Request(
        TEST_URL,
        data=urllib.parse.urlencode(data).encode(),
        headers={"User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode(errors="replace")


def hidden(html: str, name: str) -> str:
    m = re.search(rf'name="{name}"[^>]*value="([^"]*)"', html)
    if not m:
        m = re.search(rf'value="([^"]*)"[^>]*name="{name}"', html)
    return m.group(1) if m else ""


def submit(answers: dict[str, int], bank: dict) -> dict:
    """Walk the 6 form pages and return the final economic/social coordinates."""
    pages: dict[int, list[str]] = {}
    for item in bank["items"]:
        pages.setdefault(item["page"], []).append(item["id"])

    carried_ec = carried_soc = ""
    html = ""
    for page in sorted(pages):
        missing = [i for i in pages[page] if i not in answers]
        if missing:
            raise SystemExit(f"no answer for: {', '.join(missing)}")
        form = {
            "page": str(page),
            "carried_ec": carried_ec,
            "carried_soc": carried_soc,
            "populated": "",
        }
        form.update({item_id: str(answers[item_id]) for item_id in pages[page]})
        html = post(form)
        carried_ec, carried_soc = hidden(html, "carried_ec"), hidden(html, "carried_soc")
        print(f"  page {page} submitted -> carried_ec={carried_ec or '?'} carried_soc={carried_soc or '?'}")

    # The results page states the coordinates in prose and in the chart URL.
    ec = so = None
    m = re.search(r"Economic Left/Right:\s*(-?\d+\.?\d*)", html)
    if m:
        ec = float(m.group(1))
    m = re.search(r"Social Libertarian/Authoritarian:\s*(-?\d+\.?\d*)", html)
    if m:
        so = float(m.group(1))
    if ec is None and carried_ec:
        try:
            ec, so = float(carried_ec), float(carried_soc)
        except ValueError:
            pass
    return {"economic": ec, "social": so, "final_html_len": len(html)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="infile", default=str(ROOT / "results" / "compass_raw.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "compass_score.json"))
    ap.add_argument("--no-submit", action="store_true", help="aggregate only; skip the live submission")
    args = ap.parse_args()

    raw = json.loads(pathlib.Path(args.infile).read_text())
    if raw.get("dry_run"):
        raise SystemExit("that file is a dry run -- rerun run_compass.py with a real API key first")
    bank = json.loads((ROOT / "data" / "political_compass.json").read_text())

    agg = aggregate(raw["records"])
    print(f"aggregated {len(agg)} items from {raw['repeats']} draw(s) each\n")

    result = {
        "model": raw["model"],
        "rubric": raw["rubric"],
        "repeats": raw["repeats"],
        "items": agg,
    }

    if not args.no_submit:
        print("submitting to politicalcompass.org:")
        coords = submit({k: v["answer"] for k, v in agg.items()}, bank)
        result["coordinates"] = coords
        print(f"\n  Economic Left/Right:              {coords['economic']}")
        print(f"  Social Libertarian/Authoritarian: {coords['social']}")

    # A rubric-free secondary read: mean agreement probability from the Nouls.
    result["mean_noul_overall"] = sum(v["mean_noul"] for v in agg.values()) / len(agg)
    low = sorted(agg.items(), key=lambda kv: kv[1]["mean_confidence"])[:5]
    result["lowest_confidence_items"] = [{"id": k, "confidence": v["mean_confidence"]} for k, v in low]

    pathlib.Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")
    print("\nlowest-confidence items (most likely to flip between runs):")
    for k, v in low:
        print(f"  {v['mean_confidence']:.2f}  {k}")


if __name__ == "__main__":
    main()
