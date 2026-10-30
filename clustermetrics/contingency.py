"""Contingency-matrix based external clustering evaluation.

Every metric here is computed from the contingency matrix ``M`` where
``M[i, j]`` is the number of points of true class ``i`` assigned to predicted
cluster ``j``. Row sums ``a_i`` are class sizes, column sums ``b_j`` are
cluster sizes and ``N`` is the total number of points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Contingency:
    matrix: np.ndarray
    classes: np.ndarray
    clusters: np.ndarray

    @property
    def n(self) -> int:
        return int(self.matrix.sum())

    @property
    def class_sizes(self) -> np.ndarray:
        return self.matrix.sum(axis=1)

    @property
    def cluster_sizes(self) -> np.ndarray:
        return self.matrix.sum(axis=0)

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            self.matrix,
            index=[f"class {c}" for c in self.classes],
            columns=[f"cluster {k}" for k in self.clusters],
        )


def contingency_matrix(y_true, y_pred) -> Contingency:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D arrays of equal length")
    if y_true.size == 0:
        raise ValueError("cannot evaluate an empty labelling")

    classes, class_idx = np.unique(y_true, return_inverse=True)
    clusters, cluster_idx = np.unique(y_pred, return_inverse=True)
    matrix = np.zeros((classes.size, clusters.size), dtype=np.int64)
    np.add.at(matrix, (class_idx, cluster_idx), 1)
    return Contingency(matrix, classes, clusters)


def _as_contingency(y_true_or_c, y_pred=None) -> Contingency:
    if isinstance(y_true_or_c, Contingency):
        return y_true_or_c
    return contingency_matrix(y_true_or_c, y_pred)


def _comb2(x) -> np.ndarray | float:
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1.0) / 2.0


def _safe_div(num: float, den: float, empty: float) -> float:
    return float(num / den) if den > 0 else empty


def _f(p: float, r: float) -> float:
    return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0


# --- Set-matching metrics ---------------------------------------------------


def purity(y_true, y_pred=None) -> float:
    """Fraction of points belonging to the majority class of their cluster."""
    c = _as_contingency(y_true, y_pred)
    return float(c.matrix.max(axis=0).sum() / c.n)


def inverse_purity(y_true, y_pred=None) -> float:
    """Fraction of points belonging to the dominant cluster of their class."""
    c = _as_contingency(y_true, y_pred)
    return float(c.matrix.max(axis=1).sum() / c.n)


def max_matching_accuracy(y_true, y_pred=None) -> tuple[float, dict]:
    """Accuracy under the best one-to-one class/cluster mapping (Hungarian).

    When the number of clusters differs from the number of classes, surplus
    clusters (or classes) stay unmatched and all their points count as errors.
    Returns the accuracy and the ``{cluster: class}`` mapping.
    """
    c = _as_contingency(y_true, y_pred)
    rows, cols = linear_sum_assignment(c.matrix, maximize=True)
    mapping = {c.clusters[j].item(): c.classes[i].item() for i, j in zip(rows, cols)}
    return float(c.matrix[rows, cols].sum() / c.n), mapping


def class_f_measure(y_true, y_pred=None) -> float:
    """Larsen & Aone (1999) F-measure.

    For each class, take the best F1 over all clusters, then average weighted
    by class size. A cluster may be the best match for several classes.
    """
    c = _as_contingency(y_true, y_pred)
    m = c.matrix.astype(np.float64)
    a = c.class_sizes[:, None].astype(np.float64)
    b = c.cluster_sizes[None, :].astype(np.float64)
    precision = m / b
    recall = m / a
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(m > 0, 2 * precision * recall / (precision + recall), 0.0)
    return float((c.class_sizes * f.max(axis=1)).sum() / c.n)


# --- Pair-counting metrics --------------------------------------------------


def pair_counts(y_true, y_pred=None) -> dict[str, float]:
    """Classify all N(N-1)/2 point pairs.

    TP: same class & same cluster; FP: different class, same cluster;
    FN: same class, different cluster; TN: different class & different cluster.
    """
    c = _as_contingency(y_true, y_pred)
    same_both = float(_comb2(c.matrix).sum())
    same_cluster = float(_comb2(c.cluster_sizes).sum())
    same_class = float(_comb2(c.class_sizes).sum())
    total = float(_comb2(c.n))
    tp = same_both
    fp = same_cluster - tp
    fn = same_class - tp
    tn = total - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def pair_precision_recall_f(y_true, y_pred=None) -> tuple[float, float, float]:
    """Pairwise precision, recall and F1.

    Precision: of pairs put together, the fraction that share a class.
    Recall: of pairs sharing a class, the fraction put together.
    0/0 precision (all singletons) is defined as 1 since no wrong pair is
    asserted; 0/0 recall (all classes singletons) is defined likewise.
    """
    pc = pair_counts(y_true, y_pred)
    p = _safe_div(pc["tp"], pc["tp"] + pc["fp"], 1.0)
    r = _safe_div(pc["tp"], pc["tp"] + pc["fn"], 1.0)
    return p, r, _f(p, r)


def bcubed(y_true, y_pred=None) -> tuple[float, float, float]:
    """BCubed precision, recall and F1 (Bagga & Baldwin 1998; Amigo et al. 2009)."""
    c = _as_contingency(y_true, y_pred)
    m = c.matrix.astype(np.float64)
    sq = m**2
    p = float((sq / c.cluster_sizes[None, :]).sum() / c.n)
    r = float((sq / c.class_sizes[:, None]).sum() / c.n)
    return p, r, _f(p, r)


def adjusted_rand_index(y_true, y_pred=None) -> float:
    c = _as_contingency(y_true, y_pred)
    index = float(_comb2(c.matrix).sum())
    sum_a = float(_comb2(c.class_sizes).sum())
    sum_b = float(_comb2(c.cluster_sizes).sum())
    expected = sum_a * sum_b / float(_comb2(c.n)) if c.n > 1 else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    if np.isclose(max_index, expected):
        return 1.0
    return (index - expected) / (max_index - expected)


# --- Information-theoretic --------------------------------------------------


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def normalized_mutual_info(y_true, y_pred=None) -> float:
    """NMI with arithmetic-mean normalisation (sklearn's default)."""
    c = _as_contingency(y_true, y_pred)
    if c.classes.size == c.clusters.size == 1:
        return 1.0
    m = c.matrix.astype(np.float64)
    n = c.n
    nz = m > 0
    outer = np.outer(c.class_sizes, c.cluster_sizes).astype(np.float64)
    mi = float((m[nz] / n * np.log(m[nz] * n / outer[nz])).sum())
    h_true = _entropy(c.class_sizes.astype(np.float64))
    h_pred = _entropy(c.cluster_sizes.astype(np.float64))
    denom = 0.5 * (h_true + h_pred)
    if denom <= 0:
        return 1.0
    return max(mi, 0.0) / denom


# --- Bundle ------------------------------------------------------------------

METRIC_LABELS = {
    "purity": "Purity",
    "inverse_purity": "Inverse purity",
    "accuracy": "Max-matching accuracy",
    "pair_precision": "Pair precision",
    "pair_recall": "Pair recall",
    "pair_f1": "Pair F1",
    "class_f": "Class F-measure",
    "bcubed_precision": "BCubed precision",
    "bcubed_recall": "BCubed recall",
    "bcubed_f1": "BCubed F1",
    "ari": "ARI",
    "nmi": "NMI",
}


def evaluate(y_true, y_pred=None) -> dict[str, float]:
    """All metrics for one labelling; higher is better for every entry."""
    c = _as_contingency(y_true, y_pred)
    pp, pr, pf = pair_precision_recall_f(c)
    bp, br, bf = bcubed(c)
    return {
        "purity": purity(c),
        "inverse_purity": inverse_purity(c),
        "accuracy": max_matching_accuracy(c)[0],
        "pair_precision": pp,
        "pair_recall": pr,
        "pair_f1": pf,
        "class_f": class_f_measure(c),
        "bcubed_precision": bp,
        "bcubed_recall": br,
        "bcubed_f1": bf,
        "ari": adjusted_rand_index(c),
        "nmi": normalized_mutual_info(c),
        "n_clusters": int(c.clusters.size),
    }
