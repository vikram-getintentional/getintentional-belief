import numpy as np
from dataclasses import dataclass
from .states import BeliefState, STATES

# Utility

def row_normalize(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    rs = A.sum(axis=1, keepdims=True)
    rs[rs == 0.0] = 1.0
    return A / rs

@dataclass
class LogisticCoeffs:
    bias: float
    e: float  # evidence coefficient
    g: float  # graph (out-strength) coefficient
    r: float  # relevance coefficient

DEFAULT_THETA = {
    "S0_S1": LogisticCoeffs(bias=-1.0, e=3.0, g=1.0, r=0.5),   # Unaware -> ProblemRealisation
    "S1_S2": LogisticCoeffs(bias=-0.5, e=2.5, g=1.0, r=0.5),   # ProblemRealisation -> PainRealisation
    "S2_S3": LogisticCoeffs(bias=-0.8, e=2.0, g=0.6, r=0.8),   # PainRealisation -> Discovery
    "S3_S4": LogisticCoeffs(bias=-1.2, e=2.0, g=0.5, r=1.0),   # Discovery -> Barriers
    "S4_S5": LogisticCoeffs(bias=-1.2, e=1.8, g=0.4, r=1.2),   # Barriers -> Implementation
}

def ProbBeliefMachineUniform(A, beta=0.35):
    n = A.shape[0]
    r = np.ones(n)
    return ProbBeliefMachine(A, r, beta)

class ProbBeliefMachine:
    """
    Probabilistic belief state machine with per-persona transition matrices.
    Uses a single adjacency A for both influence (row-normalized) and graph features (out-strength).
    """
    def __init__(self, A: np.ndarray, r: np.ndarray, beta: float = 0.3, theta: dict | None = None):
        A = np.asarray(A, dtype=float)
        r = np.asarray(r, dtype=float)
        assert A.shape[0] == A.shape[1], "A must be square (n x n personas)."
        assert A.shape[0] == r.shape[0], "r length must match A size."
        self.A = A
        self.n = A.shape[0]
        self.r = r / (r.sum() if r.sum() != 0 else 1.0)
        self.M = row_normalize(A)
        I = np.eye(self.n)
        self.T = np.linalg.inv(I - beta * self.M)
        self.p = self.T.T @ self.r  # global sensitivity per persona
        out = A.sum(axis=1)
        mx = out.max() if out.size else 1.0
        self.G = out / (mx if mx != 0 else 1.0)  # graph broadcast proxy in [0,1]
        # Initial state distributions: everyone Unaware
        self.pi = np.zeros((self.n, len(STATES)))
        self.pi[:, BeliefState.Unaware] = 1.0
        # Logistic coeffs per forward edge
        self.theta = theta if theta is not None else DEFAULT_THETA

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _p_forward(self, i: int, key: str, evidence_vec: np.ndarray, coeffs: LogisticCoeffs) -> float:
        x = coeffs.bias + coeffs.e * evidence_vec[i] + coeffs.g * self.G[i] + coeffs.r * self.r[i]
        return float(self._sigmoid(x))

    def step(self, E: dict[str, np.ndarray]) -> None:
        """
        Advance the machine one tick with evidence dict E mapping stage keys to per-persona evidence in [0,1].
        Expected keys:
          E["PR_ZMOT"], E["PR_METRICS"], E["DISC"], E["BARR"], E["IMPL"]
        """
        n, S = self.n, len(STATES)
        P = np.zeros((n, S, S))
        # Build per-persona transition matrices
        for i in range(n):
            p01 = self._p_forward(i, "S0_S1", E["PR_ZMOT"],   self.theta["S0_S1"])  # Unaware->ProblemRealisation
            p12 = self._p_forward(i, "S1_S2", E["PR_METRICS"], self.theta["S1_S2"])  # ProblemRealisation->PainRealisation
            p23 = self._p_forward(i, "S2_S3", E["DISC"],       self.theta["S2_S3"])  # PainRealisation->Discovery
            p34 = self._p_forward(i, "S3_S4", E["BARR"],       self.theta["S3_S4"])  # Discovery->Barriers
            p45 = self._p_forward(i, "S4_S5", E["IMPL"],       self.theta["S4_S5"])  # Barriers->Implementation

            # Start with identity (self-loops), then overwrite forward transitions
            P[i] = np.eye(S)
            P[i, BeliefState.Unaware,            BeliefState.Unaware]            = 1 - p01
            P[i, BeliefState.Unaware,            BeliefState.ProblemRealisation] = p01
            P[i, BeliefState.ProblemRealisation, BeliefState.ProblemRealisation] = 1 - p12
            P[i, BeliefState.ProblemRealisation, BeliefState.PainRealisation]    = p12
            P[i, BeliefState.PainRealisation,    BeliefState.PainRealisation]    = 1 - p23
            P[i, BeliefState.PainRealisation,    BeliefState.Discovery]          = p23
            P[i, BeliefState.Discovery,          BeliefState.Discovery]          = 1 - p34
            P[i, BeliefState.Discovery,          BeliefState.Barriers]           = p34
            P[i, BeliefState.Barriers,           BeliefState.Barriers]           = 1 - p45
            P[i, BeliefState.Barriers,           BeliefState.Implementation]     = p45
            # Implementation is absorbing in this starter

        # π <- π P  (batched for all personas)
        self.pi = np.einsum('isk,ik->is', P, self.pi)

    # --- Activations (expected belief above thresholds) ---
    def activation(self, mode: str = "SOL") -> np.ndarray:
        # PR activation = prob at/above PainRealisation (S2+)
        # SOL activation = prob at/above Discovery (S3+)
        k = BeliefState.PainRealisation if mode == "PR" else BeliefState.Discovery
        return self.pi[:, k:].sum(axis=1)

    # --- Global score ---
    def expected_score(self, mode: str = "SOL") -> float:
        b = self.activation(mode)
        return float(self.r @ (self.T @ b))  # S = r^T T b

    # --- Marginal lift for persona i if we force a transition this tick ---
    def forced_transition(self, i: int, from_state: BeliefState, to_state: BeliefState) -> None:
        S = len(STATES)
        row = np.zeros((S,))
        row[to_state] = 1.0
        P = np.eye(S)
        P[from_state, :] = row
        self.pi[i, :] = self.pi[i, :] @ P

    def marginal_lift_if_forced(self, i: int, from_state: BeliefState, to_state: BeliefState, mode: str = "SOL") -> float:
        before = self.expected_score(mode)
        saved = self.pi[i, :].copy()
        self.forced_transition(i, from_state, to_state)
        after = self.expected_score(mode)
        self.pi[i, :] = saved  # revert
        return after - before