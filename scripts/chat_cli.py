from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factcheck.factory import build_agent
from factcheck.observability import flush


def main() -> None:
    agent = build_agent()
    session_id = uuid.uuid4().hex
    print("Claim checker ready. Type a claim, or 'quit' to exit.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            break
        reply = agent.ask(session_id, text)
        print(f"\n{reply.text}")
        if reply.result:
            timings = ", ".join(f"{k}={v}s" for k, v in reply.result.timings.items())
            print(f"\n[score={reply.result.top_score:.3f} attempts={reply.result.attempts} {timings}]")
    flush()


if __name__ == "__main__":
    main()
