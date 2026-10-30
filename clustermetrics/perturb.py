"""Controlled corruptions of a ground-truth labelling.

Each function returns a new predicted labelling so that the exact kind and
amount of error is known, which lets us attribute metric behaviour to a
specific failure mode.
"""

from __future__ import annotations

import numpy as np


def _relabel(y) -> np.ndarray:
    return np.unique(np.asarray(y), return_inverse=True)[1]


def fragment(y_true, classes, pieces: int, rng=None) -> np.ndarray:
    """Split every class in ``classes`` into ``pieces`` near-equal clusters."""
    rng = np.random.default_rng(rng)
    y = _relabel(y_true).astype(np.int64)
    pred = y.copy()
    next_label = int(y.max()) + 1
    for cls in np.atleast_1d(classes):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for part in np.array_split(idx, pieces)[1:]:
            pred[part] = next_label
            next_label += 1
    return _relabel(pred)


def fragment_all(y_true, pieces: int, rng=None) -> np.ndarray:
    return fragment(y_true, np.unique(_relabel(y_true)), pieces, rng)


def merge(y_true, groups) -> np.ndarray:
    """Merge classes; ``groups`` is an iterable of class-id collections."""
    y = _relabel(y_true)
    pred = y.copy()
    for group in groups:
        group = list(group)
        pred[np.isin(y, group)] = group[0]
    return _relabel(pred)


def label_noise(y_true, rate: float, rng=None) -> np.ndarray:
    """Reassign a ``rate`` fraction of points to a uniformly random other label."""
    rng = np.random.default_rng(rng)
    y = _relabel(y_true)
    k = int(y.max()) + 1
    pred = y.copy()
    flip = rng.random(y.size) < rate
    shift = rng.integers(1, k, size=flip.sum()) if k > 1 else 0
    pred[flip] = (y[flip] + shift) % max(k, 1)
    return pred


def single_cluster(y_true) -> np.ndarray:
    return np.zeros(len(y_true), dtype=np.int64)


def singletons(y_true) -> np.ndarray:
    return np.arange(len(y_true), dtype=np.int64)
