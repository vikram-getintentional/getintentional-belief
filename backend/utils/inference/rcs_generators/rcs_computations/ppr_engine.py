# ============================
# File: backend/utils/graph_base/ppr_engine.py
# ============================
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Iterable, Optional


import numpy as np
import scipy.sparse as sp




@dataclass
class PPREngine:
    """
    Reusable Personalized PageRank engine on a fixed, reversed, row-stochastic
    transition matrix P (CSR). Supports fast 'overlay' boosts by scaling specific
    edges, row-renormalizing only the touched rows, and reusing warm starts.
    """
    n: int
    node_index: Dict[str, int] # node_id -> idx
    idx_node: List[str] # idx -> node_id
    P_base: sp.csr_matrix # reversed graph transition (row-stochastic)
    alpha: float = 0.85


    # caches/CSR internals
    _pi_cache: Dict[Tuple[str, str], np.ndarray] = None
    _rowptr: np.ndarray = None
    _colind: np.ndarray = None
    _data_base: np.ndarray = None
    _edge_pos: Dict[Tuple[int, int], int] = None # (row, col) -> data index in CSR


    def __post_init__(self):
        self._pi_cache = {}
        self._rowptr = self.P_base.indptr
        self._colind = self.P_base.indices
        self._data_base = self.P_base.data
        # Precompute fast lookup from (row, col) -> pos
        self._edge_pos = {}
        for i in range(self.n):
            start, end = self._rowptr[i], self._rowptr[i + 1]
            cols = self._colind[start:end]
            for off, j in enumerate(cols):
                self._edge_pos[(i, j)] = start + off


    # ---------- Basics ----------
    def seeds_vec(self, seeds: List[str]) -> np.ndarray:
        v = np.zeros(self.n, dtype=float)
        valid = [self.node_index[s] for s in (seeds or []) if s in self.node_index]
        if valid:
            v[valid] = 1.0 / len(valid)
        else:
            v[:] = 1.0 / self.n
        return v


    # ---------- Overlays ----------
    def _apply_overlay(self, edge_scales: Dict[Tuple[int, int], float]) -> sp.csr_matrix:
        if not edge_scales:
            return self.P_base
        data = self._data_base.copy()
        touched_rows = set()
        for (i, j), k in edge_scales.items():
            pos = self._edge_pos.get((i, j))
            if pos is None:
                continue
            data[pos] = data[pos] * float(k)
            touched_rows.add(i)
        # Row re-normalize touched rows
        for i in touched_rows:
            start, end = self._rowptr[i], self._rowptr[i + 1]
            s = data[start:end].sum()
            if s > 0:
                data[start:end] /= s
        return sp.csr_matrix((data, self._colind, self._rowptr), shape=self.P_base.shape)


    def _pagerank(self, P: sp.csr_matrix, s: np.ndarray, warm: Optional[np.ndarray]) -> np.ndarray:
        alpha = float(self.alpha)
        pi = warm.copy() if warm is not None else s.copy()
        tele = (1.0 - alpha) * s
        for _ in range(200):
            new = alpha * (P.T @ pi) + tele
            if np.linalg.norm(new - pi, 1) < 1e-10:
                out = new / (new.sum() or 1.0)
                return out
        pi = new
        return pi

    def pr(
        self,
        seeds: List[str],
        *,
        boost_key: str = "",
        edge_scales: Optional[Dict[Tuple[int, int], float]] = None,
        warm_from: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, float]:
        seeds_key = "|".join(sorted(seeds or [])) or "uniform"
        cache_key = (boost_key, seeds_key)
        if cache_key in self._pi_cache:
            pi = self._pi_cache[cache_key]
        else:
            s = self.seeds_vec(seeds)
            P = self._apply_overlay(edge_scales)
            warm = self._pi_cache.get(warm_from) if warm_from else None
            pi = self._pagerank(P, s, warm)
            self._pi_cache[cache_key] = pi
        return {self.idx_node[i]: float(v) for i, v in enumerate(pi)}


    def pr_raw(
        self,
        seeds: List[str],
        *,
        boost_key: str = "",
        edge_scales: Optional[Dict[Tuple[int, int], float]] = None,
        warm_from: Optional[Tuple[str, str]] = None,
    ) -> np.ndarray:
        """Same as pr(), but returns the raw numpy vector (for chaining)."""
        seeds_key = "|".join(sorted(seeds or [])) or "uniform"
        cache_key = (boost_key, seeds_key)
        if cache_key in self._pi_cache:
            return self._pi_cache[cache_key]
        s = self.seeds_vec(seeds)
        P = self._apply_overlay(edge_scales)
        warm = self._pi_cache.get(warm_from) if warm_from else None
        pi = self._pagerank(P, s, warm)
        
        self._pi_cache[cache_key] = pi
        return pi