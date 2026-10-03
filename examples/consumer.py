"""A consumer: an app that calls llama-server, taking a lease on the model it uses.

    python examples/consumer.py "Hold it right there, fluffball."

The lease is what tells the arbiter the model is busy (so it is not evicted mid-reply) and what
loads it when it is not resident. Run llama-server with --no-models-autoload so that a request
without a lease fails loudly instead of loading a model behind the arbiter's back.
"""

from __future__ import annotations

import json
import sys
import urllib.request

import vrambiter

LLAMA = "http://127.0.0.1:8000"
MODEL_ID = "llama/gemma-4-26b"  # vrambiter's id: <satellite>/<model> from vrambiter.toml
ROUTER_MODEL = "gemma-4-26b"  # llama-server's own name for it (router_model in the config)


def chat(prompt: str) -> str:
    request = urllib.request.Request(
        f"{LLAMA}/v1/chat/completions",
        data=json.dumps(
            {"model": ROUTER_MODEL, "messages": [{"role": "user", "content": prompt}]}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def main() -> None:
    arb = vrambiter.connect("chat-app")
    try:
        # Waits for room (evicting idle models elsewhere if needed) and for the load.
        with arb.lease(MODEL_ID, timeout=300):
            print(chat(" ".join(sys.argv[1:]) or "Say hello."))
    except vrambiter.VramUnavailable as exc:
        print(f"no room for {MODEL_ID}: {exc}", file=sys.stderr)
        for holder in exc.holders:
            print(f"  {holder.describe()}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
