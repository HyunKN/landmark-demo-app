"""JSONL 디버그 로그."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DebugLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, kind: str, input_id: str, elapsed_ms: int, below_threshold: bool, top3: list[dict], scores: dict, extra: dict[str, Any] | None = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "input_id": input_id,
            "elapsed_ms": elapsed_ms,
            "below_threshold": below_threshold,
            "top3": top3,
            "scores": scores,
        }
        if extra:
            entry.update(extra)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
