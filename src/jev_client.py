"""Minimal client for TypeSafe's Jev "System One" model.

Two providers are supported and speak the same {model, state, questions} body:

  openrouter  POST https://openrouter.ai/api/alpha/decisions   model typesafe/jev-1.13
  typesafe    POST https://api.typesafe.ai/v1/systemone        model jev-latest

OpenRouter is the default because it is self-serve; the direct TypeSafe API is
waitlisted. Note that Jev is NOT reachable via chat/completions on either host --
it only answers on the decisions endpoint.

Docs: https://docs.typesafe.ai/api
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

PROVIDERS = {
    "openrouter": {
        "url": "https://openrouter.ai/api/alpha/decisions",
        "model": "typesafe/jev-1.13",
        "key_env": "OPENROUTER_API_KEY",
    },
    "typesafe": {
        "url": "https://api.typesafe.ai/v1/systemone",
        "model": "jev-latest",
        "key_env": "TYPESAFE_API_KEY",
    },
}

# Published limits: 250k tokens/sec, 1,200 requests/min.
# 520/522/524 are Cloudflare edge errors seen in practice on the OpenRouter route.
RETRY_STATUSES = {429, 500, 502, 503, 520, 522, 524, 529}


def choice(instructions: str, criteria: dict[str, str | None]) -> dict:
    """A Choice question: pick one of up to 255 options."""
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, criteria: list[str]) -> dict:
    """A Score question: position on an ordered rubric of 2-10 levels."""
    if not 2 <= len(criteria) <= 10:
        raise ValueError(f"Score needs 2-10 levels, got {len(criteria)}")
    return {"type": "score", "instructions": instructions, "criteria": criteria}


def noul(instructions: str, criteria: dict[str, str] | None = None) -> dict:
    """A Noul question: is this statement true? Returns a 0-1 probability."""
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria:
        q["criteria"] = criteria
    return q


@dataclass
class JevClient:
    provider: str = "openrouter"
    api_key: str | None = None
    model: str | None = None
    timeout: float = 60.0
    max_retries: int = 5
    dry_run: bool = False
    # populated as we go, so callers can report spend
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    calls: int = 0

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}; pick one of {list(PROVIDERS)}")
        cfg = PROVIDERS[self.provider]
        self.url = cfg["url"]
        self.model = self.model or cfg["model"]
        self.key_env = cfg["key_env"]
        if not self.dry_run:
            self.api_key = self.api_key or os.environ.get(self.key_env)
            if not self.api_key:
                raise SystemExit(
                    f"No API key. Set {self.key_env}, or pass --dry-run to inspect "
                    f"the request payloads without calling the API."
                )

    def ask(self, state: Any, questions: dict[str, dict]) -> dict:
        """Send one decisions request. Questions are answered in parallel server-side."""
        body = {"model": self.model, "state": state, "questions": questions}
        if self.dry_run:
            return {"_dry_run": True, "request": body}

        payload = json.dumps(body).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            # Optional attribution headers OpenRouter uses for its rankings.
            headers["X-Title"] = "jev-political-benchmarks"

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read())
                self.calls += 1
                for k in ("input_tokens", "output_tokens"):
                    self.usage[k] += (out.get("usage") or {}).get(k, 0)
                return out
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:400]
                if e.code not in RETRY_STATUSES:
                    raise RuntimeError(f"HTTP {e.code} from {self.url}: {detail}") from e
                last_err = RuntimeError(f"HTTP {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
            # exponential backoff with jitter
            if attempt < self.max_retries - 1:
                time.sleep(min(2**attempt + random.random(), 30))
        raise RuntimeError(f"giving up after {self.max_retries} attempts: {last_err}")

    def cost_usd(self) -> float:
        """Input tokens are $0.042/M; output tokens are free."""
        return self.usage["input_tokens"] / 1_000_000 * 0.042
