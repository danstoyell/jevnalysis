"""Make Jev answer a question in free text, one character at a time.

Jev emits no free text -- it only picks among typed options with calibrated
probabilities. So we build a decoder around it: at every step the model is shown
the question plus the answer built so far, and asked a single `Choice` question
over 28 options -- the 26 lowercase letters, a space, and STOP. The pick is
appended and the loop runs again. That is exactly autoregressive decoding, with
a character alphabet and Jev's decision endpoint standing in for a softmax over
tokens, which turns a System One decision model into a (slow, basic, lowercase)
classic LLM.

Sampling mirrors what a chat model does with logits. Jev's probabilities are
already calibrated, so by default we sample from them directly at temperature
1.0; --greedy takes the model's own top pick instead, and --temperature /
--top-p reshape the distribution the usual way. Note Jev rounds probabilities to
two decimals, so the tail below 0.005 is not merely unlikely, it is unreachable.

Two masks are applied before sampling, both to head off degenerate output that
has nothing to do with the model's opinion: space cannot be picked at the start
or straight after another space, and STOP cannot be picked before --min-chars.

Caveats worth keeping in mind before reading anything into the output:

  * Every step is an independent request. Jev has no notion of a running
    generation, and the docs warn that "accuracy falls as the state grows with
    content unrelated to the decision" -- here the state grows with the answer
    itself, which is related, but it does grow.
  * Character-level prediction is not what a classifier is built for, and
    `docs.typesafe.ai/model-jaggedness/jev-1.13` makes no promise about it.
    Expect the output to read like a small character-level RNN, not like chat.
  * One call per character. A 200-character answer is 200 calls; at Jev's
    $0.042/M input tokens that is still well under a cent, but it is not fast.

Usage:
    python ask_jev.py "what is the capital of france"
    python ask_jev.py "describe the weather" --greedy --max-chars 120
    python ask_jev.py "hello" --dry-run          # print the first payload, send nothing
    python ask_jev.py "hello" --show-probs --trace out.json

Docs: https://docs.typesafe.ai/api - https://docs.typesafe.ai/primitives/choice
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import string
import sys

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from jev_client import JevClient, choice  # noqa: E402

LETTERS = list(string.ascii_lowercase)
SPACE = "space"
STOP = "STOP"
ALPHABET = LETTERS + [SPACE, STOP]

INSTRUCTIONS = (
    "The state gives a question and the answer written so far, character by character. "
    "Select the single character that comes next in the answer. "
    "Select STOP only when the answer is already complete as written."
)

# Two criteria styles, the same split run_compass.py makes between --rubric plain
# and --rubric anchored: bare labels, or each option spelled out as a situation,
# which is what TypeSafe's docs ask for.
CRITERIA_PLAIN: dict[str, str | None] = {k: None for k in ALPHABET}
CRITERIA_DESCRIBED: dict[str, str | None] = {
    **{c: f"The answer continues with the letter {c!r}." for c in LETTERS},
    SPACE: "The current word is finished and the answer continues with a space, then another word.",
    STOP: "The answer is finished. Nothing further should be appended to it.",
}


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


def legal_options(answer: str, min_chars: int) -> list[str]:
    """Mask the options that would only produce degenerate text."""
    opts = list(LETTERS)
    if answer and not answer.endswith(" "):
        opts.append(SPACE)
    if len(answer) >= min_chars:
        opts.append(STOP)
    return opts


def pick(
    probs: dict[str, float],
    model_choice: str,
    legal: list[str],
    rng: random.Random,
    greedy: bool,
    temperature: float,
    top_p: float,
) -> str:
    """Turn Jev's distribution into one character, the way a sampler turns logits
    into one token."""
    weights = {k: max(probs.get(k, 0.0), 0.0) for k in legal}
    total = sum(weights.values())

    if total <= 0:
        # Jev put everything on masked options (or rounded the rest to zero).
        # Fall back to its own pick if that is legal, else spread uniformly.
        if model_choice in legal:
            return model_choice
        weights = {k: 1.0 for k in legal}
        total = float(len(legal))

    if greedy:
        return max(weights, key=lambda k: (weights[k], k))

    if temperature != 1.0:
        t = max(temperature, 1e-6)
        weights = {k: (v / total) ** (1.0 / t) for k, v in weights.items()}
        total = sum(weights.values())

    ranked = sorted(weights.items(), key=lambda kv: -kv[1])
    if top_p < 1.0:
        kept, acc = [], 0.0
        for k, v in ranked:
            kept.append((k, v))
            acc += v / total
            if acc >= top_p - 1e-9:  # tolerate float drift at the cutoff
                break
        ranked = kept
        total = sum(v for _, v in ranked)

    r = rng.random() * total
    for k, v in ranked:
        r -= v
        if r <= 0:
            return k
    return ranked[-1][0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("question", help="the question to put to Jev")
    ap.add_argument("--provider", default="openrouter", choices=["openrouter", "typesafe"])
    ap.add_argument("--max-chars", type=int, default=200, help="hard cap on answer length")
    ap.add_argument("--min-chars", type=int, default=1, help="STOP is masked below this length")
    ap.add_argument("--greedy", action="store_true", help="take the top option instead of sampling")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--criteria", default="described", choices=["described", "plain"],
                    help="spell each option out as a situation, or show bare labels")
    ap.add_argument("--show-probs", action="store_true", help="per-step top options, to stderr")
    ap.add_argument("--trace", help="write the full per-step distributions to this JSON file")
    ap.add_argument("--dry-run", action="store_true", help="print the first payload, send nothing")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    criteria = CRITERIA_DESCRIBED if args.criteria == "described" else CRITERIA_PLAIN
    question = {"next_character": choice(INSTRUCTIONS, criteria)}
    client = JevClient(provider=args.provider, dry_run=args.dry_run)
    rng = random.Random(args.seed)

    answer = ""
    steps: list[dict] = []
    stopped = "max_chars"

    try:
        while len(answer) < args.max_chars:
            state = {"question": args.question, "answer_so_far": answer}
            resp = client.ask(state, question)

            if args.dry_run:
                print(json.dumps(resp["request"], indent=2, ensure_ascii=False))
                print(f"\nDry run: 1 of up to {args.max_chars} requests built, 0 sent.")
                return

            a = resp["answers"]["next_character"]
            legal = legal_options(answer, args.min_chars)
            key = pick(a["probabilities"], a["choice"], legal, rng,
                       args.greedy, args.temperature, args.top_p)

            top = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:4]
            steps.append({
                "step": len(steps),
                "answer_so_far": answer,
                "model_choice": a["choice"],
                "sampled": key,
                "confidence": a["confidence"],
                "probabilities": a["probabilities"],
            })
            if args.show_probs:
                shown = "  ".join(f"{k}={p:.2f}" for k, p in top)
                print(f"[{len(steps):3d}] {key:<5} <- {shown}", file=sys.stderr, flush=True)

            if key == STOP:
                stopped = "STOP"
                break
            char = " " if key == SPACE else key
            answer += char
            if not args.show_probs:
                sys.stdout.write(char)
                sys.stdout.flush()
    except KeyboardInterrupt:
        stopped = "interrupted"

    if not args.show_probs:
        sys.stdout.write("\n")
    else:
        print(f"\n{answer}")

    mean_conf = sum(s["confidence"] for s in steps) / len(steps) if steps else 0.0
    print(
        f"\n{len(answer)} chars, {client.calls} calls, ${client.cost_usd():.4f}, "
        f"mean confidence {mean_conf:.2f}, ended on {stopped}.",
        file=sys.stderr,
    )

    if args.trace:
        pathlib.Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.trace).write_text(json.dumps({
            "question": args.question,
            "answer": answer,
            "provider": args.provider,
            "model": client.model,
            "sampling": {
                "greedy": args.greedy,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
                "criteria": args.criteria,
                "min_chars": args.min_chars,
                "max_chars": args.max_chars,
            },
            "stopped_on": stopped,
            "usage": client.usage,
            "cost_usd": round(client.cost_usd(), 6),
            "steps": steps,
        }, indent=2, ensure_ascii=False))
        print(f"Trace -> {args.trace}", file=sys.stderr)


if __name__ == "__main__":
    main()
