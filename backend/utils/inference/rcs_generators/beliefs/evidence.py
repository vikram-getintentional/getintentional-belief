import numpy as np
from dataclasses import dataclass

STAGE_KEYS = ["PR_ZMOT", "PR_METRICS", "DISC", "BARR", "IMPL"]

@dataclass
class EvidenceWindow:
    n_personas: int
    decay: float = 0.85  # exponential smoothing for rolling evidence

    def __post_init__(self):
        self.buf = {k: np.zeros(self.n_personas, dtype=float) for k in STAGE_KEYS}

    def add_event(self, stage_key: str, persona_idx: int, strength: float = 1.0):
        self.buf[stage_key][persona_idx] = np.clip(self.buf[stage_key][persona_idx] + strength, 0.0, 1.0)

    def decay_tick(self):
        for k in STAGE_KEYS:
            self.buf[k] *= self.decay

    def snapshot(self) -> dict[str, np.ndarray]:
        # Return a copy suitable for machine.step(E)
        return {k: v.copy() for k, v in self.buf.items()}