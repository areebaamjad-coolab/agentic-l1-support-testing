from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from core.agent import build_support_graph

load_dotenv()
app = FastAPI(title="Agentic L1 Support PoC", version="0.2")

graph = build_support_graph()


class ChatRequest(BaseModel):
    text: str
    thread_id: str = "api-session"
    debug: bool = False


class ChatResponse(BaseModel):
    reply: str
    intent: str | None = None
    needs_human: bool = False
    escalation: dict | None = None
    debug_decision: str | None = None


def _extract_reply(state: dict) -> str:
    msgs = state.get("chat_history", [])
    for m in reversed(msgs):
        if getattr(m, "type", "") in {"ai", "assistant"}:
            return getattr(m, "content", "") or ""
    return ""


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    cfg = {"configurable": {"thread_id": req.thread_id}}
    state = graph.invoke({"chat_history": [HumanMessage(content=req.text)]}, config=cfg)

    intent = state.get("current_intent")
    needs_human = bool(state.get("needs_human", False))
    escalation = state.get("escalation")

    debug_decision = None
    if req.debug:
        if needs_human or intent == "restricted_action":
            debug_decision = "restricted_action -> logger + escalator"
        elif intent == "technical_support":
            debug_decision = "technical_support -> knowledge_base/rag"
        else:
            debug_decision = "unknown -> clarifying_question"

    return ChatResponse(
        reply=_extract_reply(state),
        intent=intent,
        needs_human=needs_human,
        escalation=escalation,
        debug_decision=debug_decision,
    )

