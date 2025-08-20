import numpy as np
from .machine import ProbBeliefMachine
from .states import BeliefState

class CommitteeScorer:
    def __init__(self, machine: ProbBeliefMachine, mode: str = "SOL"):
        self.m = machine
        self.mode = mode

    def score(self) -> float:
        return self.m.expected_score(self.mode)

    def marginal_lift(self, persona_idx: int, from_state: BeliefState, to_state: BeliefState) -> float:
        return self.m.marginal_lift_if_forced(persona_idx, from_state, to_state, mode=self.mode)

    def rank_next_best_actions(self, candidates: list[tuple[int, BeliefState, BeliefState]], top_k: int = 5):
        lifts = []
        for (i, s_from, s_to) in candidates:
            lifts.append(((i, s_from, s_to), self.marginal_lift(i, s_from, s_to)))
        lifts.sort(key=lambda x: x[1], reverse=True)
        return lifts[:top_k]