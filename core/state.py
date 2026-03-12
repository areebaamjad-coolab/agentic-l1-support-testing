from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

Intent = Literal["technical_support", "restricted_action", "unknown"]


class SupportState(TypedDict, total=False):
    """
    Minimal, explicit state for the PoC.

    - chat_history: full conversation (persisted via MemorySaver)
    - current_intent: last classified user intent
    - needs_human: whether the flow requires human intervention
    """

    chat_history: Annotated[list[BaseMessage], add_messages]
    current_intent: Intent
    needs_human: bool
    escalation: dict

