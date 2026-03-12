from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from core.agent import build_support_graph


def _print_debug(state: dict, *, enabled: bool) -> None:
    if not enabled:
        return

    intent = state.get("current_intent", "unknown")
    needs_human = bool(state.get("needs_human", False))

    if needs_human or intent == "restricted_action":
        decision = "restricted_action -> logger + escalator"
    elif intent == "technical_support":
        decision = "technical_support -> knowledge_base"
    else:
        decision = "unknown -> clarifying_question"

    print(f"[debug] decision: {decision}")


def _maybe_speak(text: str, *, enabled: bool) -> None:
    if not enabled:
        return
    try:
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        return


def main(argv: list[str]) -> int:
    load_dotenv()

    ap = argparse.ArgumentParser(description="CRM-less Agentic L1 Support PoC (LangGraph).")
    ap.add_argument("--debug", action="store_true", help="Print routing/tool choice before reply.")
    ap.add_argument(
        "--tts",
        action="store_true",
        help="Optional text-to-speech if pyttsx3 is installed (speech-ready stub).",
    )
    ap.add_argument(
        "--thread-id",
        default="local-test-session",
        help="Conversation thread id (for in-session persistence).",
    )
    args = ap.parse_args(argv)

    app = build_support_graph()
    config = {"configurable": {"thread_id": args.thread_id}}

    print("Agentic L1 Support PoC. Type 'exit' to quit.\n")
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        state = app.invoke({"chat_history": [HumanMessage(content=user_text)]}, config=config)
        _print_debug(state, enabled=args.debug)

        messages = state.get("chat_history", [])
        assistant_text = ""
        for m in reversed(messages):
            if getattr(m, "type", "") in {"ai", "assistant"}:
                assistant_text = getattr(m, "content", "") or ""
                break

        print(f"agent> {assistant_text}\n")
        _maybe_speak(assistant_text, enabled=args.tts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

