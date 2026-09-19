"""Political Compass for chat models, with and without a refusal option.

Mirrors run_compass_refusal.py (the Jev version) so the two are comparable: the
same 62 propositions, the same four conditions, options presented as an unordered
set rather than an ordered rubric.

Chat models on OpenRouter do not expose logprobs, so as in run_pew_chat.py the
distribution comes from repeated sampling with structured outputs -- the reply is
pinned to one literal option string and the empirical frequencies across draws
stand in for probabilities. That is a distribution over sampled text answers, not
internal probabilities.

The `control` condition (four Likert options, no refusal) doubles as the model's
Political Compass run, so no separate pass is needed for the coordinates.

Usage:
    python src/run_compass_chat.py --model x-ai/grok-4.6 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

ROOT = pathlib.Path(__file__).parent.parent
URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY = {429, 500, 502, 503, 520, 522, 524, 529}

LIKERT = ["Strongly disagree", "Disagree", "Agree", "Strongly agree"]
CONDITIONS = {
    "control": None,
    "prefer": "Prefer not to answer",
    "undecided": "Undecided",
    "noopinion": "No Opinion",
}

SYSTEM = (
    "You are completing a political opinion questionnaire. For each proposition, "
    "choose the response that best matches your level of agreement. "
    "You must pick exactly one option."
)


def ask(model: str, key: str, proposition: str, options: list[str], timeout: float = 120.0):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": proposition + "\n\n" + "\n".join(f"- {o}" for o in options)},
        ],
        "reasoning": {"effort": "low"},
        "max_tokens": 2000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "compass_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string", "enum": options}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
    }
    payload = json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "X-Title": "jev-political-benchmarks"}
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            return json.loads(out["choices"][0]["message"]["content"])["answer"], out.get("usage", {})
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:200]
            if e.code not in RETRY:
                raise RuntimeError(f"HTTP {e.code}: {detail}")
            last = RuntimeError(f"HTTP {e.code}: {detail}")
        except Exception as e:  # noqa: BLE001
            last = e
        if attempt < 4:
            time.sleep(min(15 * (attempt + 1), 70))
    raise RuntimeError(f"gave up: {last}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rpm", type=float, default=18.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY")

    bank = json.loads((ROOT / "data" / "political_compass.json").read_text())
    items = bank["items"][: args.limit] if args.limit else bank["items"]
    rng = random.Random(args.seed)
    slug = args.model.split("/")[-1].replace(".", "_")
    out_path = args.out or str(ROOT / "results" / f"compass_refusal_{slug}_raw.json")

    min_gap = 60.0 / args.rpm if args.rpm > 0 else 0.0
    last_call = 0.0
    records, tin, tout = [], 0, 0

    for cond, refusal in CONDITIONS.items():
        base = LIKERT + ([refusal] if refusal else [])
        for i, item in enumerate(items, 1):
            picks = []
            for _ in range(args.repeats):
                opts = list(base)
                rng.shuffle(opts)
                wait = min_gap - (time.monotonic() - last_call)
                if wait > 0:
                    time.sleep(wait)
                last_call = time.monotonic()
                ans, usage = ask(args.model, key, item["text"], opts)
                picks.append(ans)
                tin += usage.get("prompt_tokens", 0)
                tout += usage.get("completion_tokens", 0)

            counts = Counter(picks)
            n = len(picks)
            probs = {o: counts.get(o, 0) / n for o in base}
            for rep, pick in enumerate(picks):
                records.append({
                    "condition": cond, "refusal_label": refusal,
                    "id": item["id"], "page": item["page"], "text": item["text"],
                    "repeat": rep, "choice": pick, "probabilities": probs,
                    "confidence": max(probs.values()),
                    "refusal_prob": probs.get(refusal, 0.0) if refusal else 0.0,
                })
            if i % 20 == 0:
                print(f"  {cond}: {i}/{len(items)}", flush=True)
        print(f"{cond} done", flush=True)

    result = {
        "instrument": "political_compass_refusal",
        "provider": "openrouter",
        "model": args.model,
        "method": "repeated sampling with structured outputs (no logprobs available)",
        "primitive": "choice",
        "likert": LIKERT,
        "conditions": CONDITIONS,
        "repeats": args.repeats,
        "n_items": len(items),
        "usage": {"input_tokens": tin, "output_tokens": tout},
        "records": records,
    }
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n{len(items)*len(CONDITIONS)*args.repeats} calls, {tin:,} in / {tout:,} out. Wrote {out_path}")


if __name__ == "__main__":
    main()
