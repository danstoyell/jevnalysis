"""Administer the Pew 2026 typology quiz to ordinary chat models, for comparison with Jev.

Jev returns a calibrated probability distribution per item. Chat models on
OpenRouter do NOT expose logprobs (checked: neither anthropic/claude-opus-5 nor
openai/gpt-5.6-sol lists `logprobs` in supported_parameters), so the closest
available substitute is REPEATED SAMPLING: ask the same item N times and use the
empirical answer frequencies as the distribution.

That is not the same quantity as Jev's probabilities and should not be reported
as if it were -- it is a distribution over *sampled text answers*, which is
exactly the gap "My Answer is C" (ACL Findings 2024) measures. It is, however,
much closer than a single text answer, and it feeds the same scorer.

Structured outputs pin the reply to one of the literal option strings, so nothing
depends on parsing prose. Option order is shuffled per draw to control order
effects, as in run_pew.py.

Usage:
    python src/run_pew_chat.py --model anthropic/claude-opus-5 --repeats 10
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

SYSTEM = (
    "You are completing a public opinion survey. Answer each question with the "
    "option that comes closest to your view. You must pick exactly one option."
)


def ask(model: str, key: str, question: str, options: list[str], timeout: float = 120.0) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question + "\n\n" + "\n".join(f"- {o}" for o in options)},
        ],
        "reasoning": {"effort": "low"},
        "max_tokens": 2000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "survey_answer",
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
            content = out["choices"][0]["message"]["content"]
            return json.loads(content)["answer"], out.get("usage", {})
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code not in RETRY:
                raise RuntimeError(f"HTTP {e.code}: {detail}")
            last = RuntimeError(f"HTTP {e.code}: {detail}")
        except Exception as e:  # noqa: BLE001 - transport/parse hiccups are retryable here
            last = e
        if attempt < 4:
            time.sleep(min(15 * (attempt + 1), 70))
    raise RuntimeError(f"gave up: {last}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rpm", type=float, default=18.0,
                    help="pace requests; OpenRouter caps new accounts at 20/min on some models")
    ap.add_argument("--out")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY")

    bank = json.loads((ROOT / "data" / "pew_typology.json").read_text())
    items = bank["items"][: args.limit] if args.limit else bank["items"]
    rng = random.Random(args.seed)
    out_path = args.out or str(ROOT / "results" / f"pew_{args.model.split('/')[-1].replace('.', '_')}_raw.json")

    records, tok_in, tok_out = [], 0, 0
    min_gap = 60.0 / args.rpm if args.rpm > 0 else 0.0
    last_call = 0.0
    for i, item in enumerate(items, 1):
        q = f"{item['preamble']} {item['text']}" if item.get("preamble") else item["text"]
        picks = []
        for _ in range(args.repeats):
            opts = list(item["options"])
            rng.shuffle(opts)
            wait = min_gap - (time.monotonic() - last_call)
            if wait > 0:
                time.sleep(wait)
            last_call = time.monotonic()
            answer, usage = ask(args.model, key, q, [o["text"] for o in opts])
            picks.append(answer)
            tok_in += usage.get("prompt_tokens", 0)
            tok_out += usage.get("completion_tokens", 0)

        counts = Counter(picks)
        n = len(picks)
        probs = {o["text"]: counts.get(o["text"], 0) / n for o in item["options"]}
        top = max(probs, key=lambda k: probs[k])
        chosen = {o["text"]: o for o in item["options"]}[top]

        # one record per draw, matching run_pew.py's shape so score_pew.py just works
        for rep, pick in enumerate(picks):
            p = {o["text"]: o for o in item["options"]}[pick]
            records.append({
                "internal_id": item["internal_id"],
                "category": item.get("category"),
                "question": q,
                "repeat": rep,
                "chosen_uuid": p["uuid"],
                "chosen_text": p["text"],
                "points": p["points"],
                # empirical frequency across draws, repeated on each record
                "confidence": probs[top],
                "probabilities": probs,
                "option_order": [o["uuid"] for o in item["options"]],
            })
        print(f"[{i:2d}/{len(items)}] {str(item['internal_id']):<22} {probs[top]:.0%} {top[:46]}", flush=True)

    result = {
        "instrument": "pew_typology_2026",
        "provider": "openrouter",
        "model": args.model,
        "method": "repeated sampling with structured outputs (no logprobs available)",
        "repeats": args.repeats,
        "shuffled": True,
        "n_items": len(items),
        "usage": {"input_tokens": tok_in, "output_tokens": tok_out},
        "records": records,
    }
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n{len(items)*args.repeats} calls, {tok_in:,} in / {tok_out:,} out tokens. Wrote {out_path}")


if __name__ == "__main__":
    main()
