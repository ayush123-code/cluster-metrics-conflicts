"""Tools for comparing how different metrics order the same solutions."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def rank_solutions(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank solutions (rows) per metric (columns); 1 = best, ties share min rank."""
    return scores.rank(axis=0, ascending=False, method="min").astype(int)


def kendall_matrix(scores: pd.DataFrame) -> pd.DataFrame:
    cols = scores.columns
    out = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for a, b in itertools.combinations(cols, 2):
        tau = kendalltau(scores[a], scores[b]).statistic
        out.loc[a, b] = out.loc[b, a] = tau
    return out


def disagreement_rate(scores: pd.DataFrame, tol: float = 1e-12) -> pd.DataFrame:
    """Fraction of solution pairs that two metrics order in opposite directions.

    Pairs tied under either metric are excluded, so 0 means the metrics never
    contradict each other and 1 means they always do.
    """
    cols = scores.columns
    values = scores.to_numpy()
    i, j = np.triu_indices(len(scores), k=1)
    diffs = values[i] - values[j]
    signs = np.where(np.abs(diffs) <= tol, 0, np.sign(diffs))
    out = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for a, b in itertools.combinations(range(len(cols)), 2):
        both = (signs[:, a] != 0) & (signs[:, b] != 0)
        rate = float((signs[both, a] != signs[both, b]).mean()) if both.any() else 0.0
        out.iloc[a, b] = out.iloc[b, a] = rate
    return out


def pairwise_conflicts(scores: pd.DataFrame, metric_a: str, metric_b: str, tol: float = 1e-12):
    """List solution pairs (x, y) where metric_a prefers x but metric_b prefers y."""
    rows = []
    for x, y in itertools.permutations(scores.index, 2):
        da = scores.at[x, metric_a] - scores.at[y, metric_a]
        db = scores.at[x, metric_b] - scores.at[y, metric_b]
        if da > tol and db < -tol:
            rows.append((x, y, da, db))
    return pd.DataFrame(rows, columns=["preferred_by_a", "preferred_by_b", "delta_a", "delta_b"])
