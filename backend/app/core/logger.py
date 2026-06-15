from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock


class SessionLogger:
    def __init__(self, log_dir: Path, max_items: int = 500) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "session.jsonl"
        self.items: deque[dict] = deque(maxlen=max_items)
        self._lock = Lock()

    def add(
        self,
        module: str,
        level: str,
        message: str,
        state: str,
        **extra: object,
    ) -> dict:
        item = {
            "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "module": module,
            "level": level,
            "message": message,
            "state": state,
            **extra,
        }
        line = json.dumps(item, ensure_ascii=False)
        with self._lock:
            self.items.append(item)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return item

    def latest(self, limit: int = 120) -> list[dict]:
        with self._lock:
            return list(self.items)[-limit:]
