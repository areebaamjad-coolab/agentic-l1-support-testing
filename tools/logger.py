from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_restricted_request(
    *,
    user_text: str,
    reason: str,
    category: str = "restricted_action",
    path: str = "requests_log.json",
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Append restricted-action requests for manual review.
    """

    p = Path(path)
    entry: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "user_text": user_text,
        "reason": reason,
    }
    if extra:
        entry["extra"] = extra

    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []

    data.append(entry)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

