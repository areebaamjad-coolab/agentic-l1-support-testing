from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from core.agent import build_support_graph


def main() -> int:
    load_dotenv()

    cases_path = Path("evals/cases.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    app = build_support_graph()
    passed = 0

    for i, c in enumerate(cases, start=1):
        thread_id = f"eval-{i}"
        cfg = {"configurable": {"thread_id": thread_id}}
        state = app.invoke({"chat_history": [HumanMessage(content=c["input"])]}, config=cfg)

        intent = state.get("current_intent")
        needs_human = bool(state.get("needs_human", False))

        ok = True
        if c.get("expect_intent") is not None and intent != c["expect_intent"]:
            ok = False
        if c.get("expect_needs_human") is not None and needs_human != c["expect_needs_human"]:
            ok = False

        name = c.get("name", f"case-{i}")
        print(f"{'PASS' if ok else 'FAIL'} - {name} | intent={intent} needs_human={needs_human}")
        if ok:
            passed += 1

    total = len(cases)
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

