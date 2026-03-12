from __future__ import annotations

import os
import re
from typing import Literal, TypedDict, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from core.config import SETTINGS
from core.state import SupportState
from tools.knowledge_base import search_knowledge_base
from tools.rag_kb import retrieve_from_vectorstore
from tools.logger import log_restricted_request


class _Classification(TypedDict):
    intent: Literal["technical_support", "restricted_action", "unknown"]
    reason: str


_CLASSIFIER_SYSTEM = SystemMessage(
    content=(
        "You are a strict L1 support intent classifier.\n"
        "Classify the user's latest message into exactly one intent:\n"
        "- technical_support: troubleshooting, how-to, diagnostics, product usage, errors\n"
        "- restricted_action: anything requiring account/billing changes, payments, refunds, "
        "identity verification, password resets for accounts, plan changes, cancellations, "
        "PII updates, access changes, or any action that must be performed in CRM\n"
        "- unknown: unclear / not enough info\n"
        "Return a short reason."
    )
)

_ANSWER_SYSTEM = SystemMessage(
    content=(
        "You are an L1 technical support agent.\n"
        "Constraints:\n"
        "- Keep responses concise (1-2 short paragraphs or bullets).\n"
        "- Ask at most 1 clarifying question if needed.\n"
        "- Do not claim you performed actions you cannot do.\n"
        "- Prefer concrete steps."
    )
)


def _last_user_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(model=SETTINGS.llm_model, temperature=SETTINGS.llm_temperature)


_RESTRICTED_PATTERNS = [
    r"\bbilling\b",
    r"\brefund\b",
    r"\bcancel\b",
    r"\bcharge\b",
    r"\bpayment\b",
    r"\bplan\b",
    r"\bsubscription\b",
    r"\baccount\b",
    r"\bemail\b.*\bchange\b",
    r"\bpassword\b.*\breset\b",
    r"\bupdate\b.*\bprofile\b",
]


def _heuristic_classify(text: str) -> _Classification:
    t = text.lower()
    if any(re.search(p, t) for p in _RESTRICTED_PATTERNS):
        return {"intent": "restricted_action", "reason": "Heuristic match on restricted-action keywords."}
    if len(t.split()) < 2:
        return {"intent": "unknown", "reason": "Too little information."}
    return {"intent": "technical_support", "reason": "Default to troubleshooting/help."}


def _classifier_node(state: SupportState) -> SupportState:
    user_text = _last_user_text(state.get("chat_history", []))
    result: _Classification
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        llm = _build_llm().with_structured_output(_Classification)
        result = cast(_Classification, llm.invoke([_CLASSIFIER_SYSTEM, HumanMessage(content=user_text)]))
    except Exception:
        result = _heuristic_classify(user_text)
    return {
        "current_intent": result["intent"],
        "needs_human": False,
    }


def _knowledge_search_node(state: SupportState) -> SupportState:
    user_text = _last_user_text(state.get("chat_history", []))
    retrieved = retrieve_from_vectorstore(
        query=user_text,
        vectorstore_dir=SETTINGS.vectorstore_dir,
        k=SETTINGS.rag_top_k,
    )
    hits = search_knowledge_base(user_text) if not retrieved else []

    kb_context = ""
    citations = ""
    if retrieved:
        kb_context = "\n\n---\n\n".join([c.content for c in retrieved[: SETTINGS.rag_top_k]])
        citations = "Sources: " + ", ".join(sorted({c.source for c in retrieved[: SETTINGS.rag_top_k]}))
    elif hits:
        top = hits[0]
        kb_context = f"KB Article: {top.title}\n{top.answer}"
        citations = f"Source: mock_kb:{top.title}"
    else:
        kb_context = "KB Search: No direct match found."

    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        llm = _build_llm()
        msg = llm.invoke(
            [
                _ANSWER_SYSTEM,
                HumanMessage(
                    content=(
                        "User asked:\n"
                        f"{user_text}\n\n"
                        "Use this knowledge base context if helpful:\n"
                        f"{kb_context}\n\n"
                        f"{citations}\n\n"
                        "Reply as concise L1 support."
                    )
                ),
            ]
        )
        return {"chat_history": [AIMessage(content=msg.content)]}
    except Exception:
        if retrieved:
            brief = retrieved[0].content.strip().splitlines()
            answer = "\n".join(brief[:6]).strip()
            return {"chat_history": [AIMessage(content=answer)]}
        if hits:
            return {"chat_history": [AIMessage(content=hits[0].answer)]}
        return {
            "chat_history": [
                AIMessage(
                    content=(
                        "I didn't find a direct KB match. What device/app is this, and what's the exact error message?"
                    )
                )
            ]
        }


def _escalator_node(state: SupportState) -> SupportState:
    user_text = _last_user_text(state.get("chat_history", []))
    reason = "Account/billing/requested action requires a human (CRM-less PoC)."
    log_restricted_request(user_text=user_text, reason=reason, extra={"intent": "restricted_action"})

    reply = (
        "I can't help with account or billing changes in this PoC. "
        "I've flagged this for a human to handle.\n\n"
        "If you want, tell me the technical issue you're seeing and I can troubleshoot that now."
    )
    return {
        "needs_human": True,
        "escalation": {"reason": reason, "user_text": user_text},
        "chat_history": [AIMessage(content=reply)],
    }


def _unknown_node(state: SupportState) -> SupportState:
    user_text = _last_user_text(state.get("chat_history", []))
    reply = (
        "I can help, but I need one detail: what product/system is this for, and what error or symptom do you see?"
        if user_text.strip()
        else "Tell me what you need help with (technical issue vs account/billing request)."
    )
    return {"chat_history": [AIMessage(content=reply)], "current_intent": "unknown"}


def _route_after_classification(state: SupportState) -> str:
    intent = state.get("current_intent", "unknown")
    if intent == "technical_support":
        return "knowledge_search"
    if intent == "restricted_action":
        return "escalator"
    return "unknown"


def build_support_graph():
    """
    Returns a compiled LangGraph app with in-session persistence.

    Use `thread_id` in config to keep memory per session.
    """

    g: StateGraph = StateGraph(SupportState)
    g.add_node("classifier", _classifier_node)
    g.add_node("knowledge_search", _knowledge_search_node)
    g.add_node("escalator", _escalator_node)
    g.add_node("unknown", _unknown_node)

    g.set_entry_point("classifier")
    g.add_conditional_edges(
        "classifier",
        _route_after_classification,
        {
            "knowledge_search": "knowledge_search",
            "escalator": "escalator",
            "unknown": "unknown",
        },
    )

    g.add_edge("knowledge_search", END)
    g.add_edge("escalator", END)
    g.add_edge("unknown", END)

    memory = MemorySaver()
    return g.compile(checkpointer=memory)

