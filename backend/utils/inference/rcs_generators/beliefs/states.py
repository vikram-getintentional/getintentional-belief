from enum import IntEnum

class BeliefState(IntEnum):
    Unaware = 0
    ProblemRealisation = 1
    PainRealisation = 2
    Discovery = 3
    Barriers = 4
    Implementation = 5

STATES = [
    BeliefState.Unaware,
    BeliefState.ProblemRealisation,
    BeliefState.PainRealisation,
    BeliefState.Discovery,
    BeliefState.Barriers,
    BeliefState.Implementation,
]

STATE_NAMES = [s.name for s in STATES]