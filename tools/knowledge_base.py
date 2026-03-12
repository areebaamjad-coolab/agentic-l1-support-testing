from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KBHit:
    title: str
    answer: str


_MOCK_KB: dict[str, KBHit] = {
    "reset router": KBHit(
        title="Reset a home router",
        answer=(
            "1) Power-cycle: unplug 30s, plug back in, wait 2-3 min.\n"
            "2) If still down: press and hold RESET 10-15s (factory reset).\n"
            "3) Reconnect Wi-Fi and re-enter ISP/PPPoE details if needed."
        ),
    ),
    "clear browser cache": KBHit(
        title="Clear browser cache (Chrome/Edge)",
        answer=(
            "Open Settings -> Privacy -> Clear browsing data.\n"
            "Select Cached images/files (and Cookies if needed) -> Clear."
        ),
    ),
    "flush dns": KBHit(
        title="Flush DNS (Windows)",
        answer="Open Command Prompt as admin -> run: ipconfig /flushdns",
    ),
    "vpn not connecting": KBHit(
        title="VPN not connecting",
        answer=(
            "1) Check internet works without VPN.\n"
            "2) Confirm correct server/region.\n"
            "3) Try switching protocol (WireGuard/OpenVPN).\n"
            "4) Restart device; if corporate VPN, check firewall/MFA."
        ),
    ),
}


def _normalize(q: str) -> str:
    return " ".join(q.lower().strip().split())


def search_knowledge_base(query: str) -> list[KBHit]:
    """
    Mock “RAG” search over a tiny in-memory KB.

    Returns a ranked list of hits (best-first).
    """

    q = _normalize(query)
    if not q:
        return []

    hits: list[tuple[int, KBHit]] = []
    for k, v in _MOCK_KB.items():
        score = 0
        if k in q:
            score += 3
        for token in k.split():
            if token in q:
                score += 1
        if score > 0:
            hits.append((score, v))

    hits.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in hits]

