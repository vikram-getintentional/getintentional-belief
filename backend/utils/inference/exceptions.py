from __future__ import annotations

from typing import Optional

STOP_PERSONA_ID = "persona:__STOP__"
STOP_PERSONA_SUFFIX = "__STOP__"


def is_stop_persona(persona_id: Optional[str]) -> bool:
    if not persona_id:
        return False
    if persona_id == STOP_PERSONA_ID:
        return True
    return persona_id.rsplit(":", 1)[-1] == STOP_PERSONA_SUFFIX


class InferenceHalt(Exception):
    def __init__(self, *, reason: Optional[str] = None):
        message = reason or "Inference halted"
        super().__init__(message)
        self.reason = message
