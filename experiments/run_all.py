"""Run every experiment and write figures, CSVs and results/report.md.

Usage:  python experiments/run_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

from clustermetrics import METRIC_LABELS, contingency_matrix, evaluate, max_matching_accuracy
from clustermetrics import perturb
from clustermetrics.ranking import (
    disagreement_rate,
    kendall_matrix,
    pairwise_conflicts,
    rank_solutions,
)

OUT = ROOT / "results"
FIG = OUT / "figures"
SEED = 0

CENTERS = np.array([[0, 0], [6, 0], [0, 6], [6, 6], [3, 3]], dtype=float)
CLUSTER_STD = 1.3
DATASETS = {
    "balanced": [240, 240, 240, 240, 240],
    "imbalanced": [700, 250, 120, 80, 50],
}
TRUE_K = len(CENTERS)
SINGLE_ERROR_BALANCED = [250, 250, 250, 250]
SINGLE_ERROR_IMBALANCED = [700, 150, 100, 50]

ONE_SIDED = ["purity", "inverse_purity", "pair_precision", "pair_recall"]
TWO_SIDED = ["accuracy", "pair_f1", "class_f", "bcubed_f1", "ari", "nmi"]
RANKED = ONE_SIDED + TWO_SIDED
SWEEP_METRICS = ["purity", "inverse_purity", "accuracy", "pair_f1", "class_f", "bcubed_f1", "ari", "nmi"]


# --- helpers -----------------------------------------------------------------


def label(m: str) -> str:
    return METRIC_LABELS.get(m, m)


def md_table(df: pd.DataFrame, fmt: str = "{:.3f}", index: bool = True) -> str:
    header = ([str(df.index.name or "")] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)] if index else []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("–" if np.isnan(v) else fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_dataset(sizes):
    return make_blobs(
        n_samples=sizes, centers=CENTERS, cluster_std=CLUSTER_STD, random_state=SEED
    )


def argmax_all(values: pd.Series, tol: float = 1e-9) -> list:
    return list(values.index[values >= values.max() - tol])


def heatmap(ax, data: pd.DataFrame, fmt: str, cmap: str, vmin=None, vmax=None):
    im = ax.imshow(data.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(data.shape[1]), [label(c) for c in data.columns], rotation=45, ha="right")
    ax.set_yticks(range(data.shape[0]), [label(r) for r in data.index])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.iat[i, j]
            if pd.isna(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=8)
                continue
            r, g, b, _ = im.cmap(im.norm(v))
            color = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color=color)
    return im


# --- Experiment 0: data and a worked contingency matrix ------------------------


def exp_datasets(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, name in zip(axes[:2], data):
        X, y = data[name]
        ax.scatter(X[:, 0], X[:, 1], c=y, s=5, cmap="tab10")
        ax.set_title(f"{name} ground truth, sizes {DATASETS[name]}")
    X, y = data["imbalanced"]
    pred = KMeans(TRUE_K, n_init=10, random_state=SEED).fit_predict(X)
    axes[2].scatter(X[:, 0], X[:, 1], c=pred, s=5, cmap="tab10")
    axes[2].set_title(f"imbalanced, k-means k={TRUE_K}")
    for ax in axes:
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG / "datasets.png", dpi=130)
    plt.close(fig)

    pred8 = KMeans(8, n_init=10, random_state=SEED).fit_predict(X)
    c = contingency_matrix(y, pred8)
    frame = c.to_frame()
    frame["class size"] = c.class_sizes
    frame.loc["cluster size"] = list(c.cluster_sizes) + [c.n]
    frame.loc["majority class"] = list(c.classes[c.matrix.argmax(axis=0)]) + [""]
    acc, mapping = max_matching_accuracy(c)
    scores = evaluate(c)
    return {
        "frame": frame,
        "mapping": mapping,
        "acc": acc,
        "scores": scores,
        "frame_k5": contingency_matrix(y, pred).to_frame(),
        "frame_k4": contingency_matrix(
            y, KMeans(TRUE_K - 1, n_init=10, random_state=SEED).fit_predict(X)
        ).to_frame(),
    }


# --- Experiment 1: number of clusters --------------------------------------------


def exp_vary_k(data, ks=range(1, 16)):
    rows = []
    for name, (X, y) in data.items():
        for k in ks:
            pred = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
            rows.append({"dataset": name, "k": k, **evaluate(y, pred)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp1_vary_k.csv", index=False)

    best = pd.DataFrame(index=pd.Index([label(m) for m in RANKED], name="Metric"))
    for name in data:
        sub = df[df.dataset == name].set_index("k")
        best[f"best k ({name})"] = [", ".join(map(str, argmax_all(sub[m]))) for m in RANKED]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for r, name in enumerate(data):
        sub = df[df.dataset == name]
        for c, (group, title) in enumerate([(ONE_SIDED, "one-sided"), (TWO_SIDED, "two-sided")]):
            ax = axes[r, c]
            for m in group:
                ax.plot(sub.k, sub[m], marker="o", ms=3, label=label(m))
            ax.axvline(TRUE_K, color="grey", ls="--", lw=1)
            ax.set_title(f"{name} data: {title} metrics")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="lower right" if c == 0 else "upper right")
    for ax in axes[1]:
        ax.set_xlabel("number of k-means clusters (dashed = true number of classes)")
    fig.tight_layout()
    fig.savefig(FIG / "exp1_vary_k.png", dpi=130)
    plt.close(fig)
    return df, best


# --- Experiment 2: fragmentation and merging ----------------------------------------


def exp_frag_merge():
    y = np.repeat(np.arange(8), 100)
    frag = pd.DataFrame(
        [{"pieces": m, **evaluate(y, perturb.fragment_all(y, m, rng=SEED))} for m in range(1, 9)]
    )
    merge = pd.DataFrame(
        [{"merged": g, **evaluate(y, perturb.merge(y, [range(g)]))} for g in range(1, 9)]
    )
    frag.to_csv(OUT / "exp2_fragmentation.csv", index=False)
    merge.to_csv(OUT / "exp2_merging.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, df, x, title in [
        (axes[0], frag, "pieces", "Fragmentation: every class split into m clusters"),
        (axes[1], merge, "merged", "Merging: g of the 8 classes merged into one cluster"),
    ]:
        for m in RANKED:
            ax.plot(df[x], df[m], marker="o", ms=3, label=label(m))
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "exp2_frag_merge.png", dpi=130)
    plt.close(fig)

    # Single errors of comparable size, on balanced vs imbalanced truths.
    y_bal = np.repeat(np.arange(len(SINGLE_ERROR_BALANCED)), SINGLE_ERROR_BALANCED)
    y_imb = np.repeat(np.arange(len(SINGLE_ERROR_IMBALANCED)), SINGLE_ERROR_IMBALANCED)
    scenarios = {
        "balanced: split one class in 2": (y_bal, perturb.fragment(y_bal, [0], 2, rng=SEED)),
        "balanced: merge two classes": (y_bal, perturb.merge(y_bal, [[0, 1]])),
        "imbalanced: split majority in 2": (y_imb, perturb.fragment(y_imb, [0], 2, rng=SEED)),
        "imbalanced: merge two smallest": (y_imb, perturb.merge(y_imb, [[2, 3]])),
        "imbalanced: merge majority + smallest": (y_imb, perturb.merge(y_imb, [[0, 3]])),
    }
    single = pd.DataFrame({k: evaluate(*v) for k, v in scenarios.items()}).T[RANKED]
    single.index.name = "Scenario"
    single.to_csv(OUT / "exp2_single_errors.csv")
    return frag, merge, single


# --- Experiment 3: class imbalance sweep ----------------------------------------------


def imbalanced_truth(p: float, n: int = 1000, n_minor: int = 3) -> np.ndarray:
    major = int(round(p * n))
    minors = [len(a) for a in np.array_split(np.arange(n - major), n_minor)]
    return np.repeat(np.arange(n_minor + 1), [major] + minors)


IMB_SOLUTIONS = {
    "single cluster": lambda y: perturb.single_cluster(y),
    "split majority in 2": lambda y: perturb.fragment(y, [0], 2, rng=SEED),
    "merge all minorities": lambda y: perturb.merge(y, [[1, 2, 3]]),
    "20% random noise": lambda y: perturb.label_noise(y, 0.2, rng=SEED),
}


def exp_imbalance(fractions=np.round(np.arange(0.25, 0.951, 0.05), 2)):
    rows = []
    for p in fractions:
        y = imbalanced_truth(p)
        for name, make in IMB_SOLUTIONS.items():
            rows.append({"majority_fraction": p, "solution": name, **evaluate(y, make(y))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp3_imbalance.csv", index=False)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for ax, m in zip(axes.flat, SWEEP_METRICS):
        for name in IMB_SOLUTIONS:
            sub = df[df.solution == name]
            ax.plot(sub.majority_fraction, sub[m], marker="o", ms=3, label=name)
        ax.set_title(label(m))
        ax.set_ylim(-0.05, 1.02)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("majority class fraction")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "exp3_imbalance.png", dpi=130)
    plt.close(fig)

    # Where does the degenerate single cluster start beating a near-perfect split?
    cross = {}
    for m in SWEEP_METRICS:
        a = df[df.solution == "single cluster"].set_index("majority_fraction")[m]
        b = df[df.solution == "split majority in 2"].set_index("majority_fraction")[m]
        wins = (a - b) >= -1e-12
        if wins.all():
            cross[label(m)] = "always"
        elif not wins.any():
            cross[label(m)] = "never"
        else:
            first = wins[wins].index.min()
            prev = wins.index[wins.index < first].max()
            cross[label(m)] = f"({prev:.2f}, {first:.2f}]"
    closed = {label("accuracy"): "p = 2/3 (single cluster scores p, split scores 1 − p/2)"}
    cross = pd.DataFrame(
        {
            "crossover p lies in (sweep step 0.05)": cross,
            "exact crossover": {k: closed.get(k, "–") for k in cross},
        }
    )
    cross.index.name = "Metric"

    winners = {}
    for p in [0.25, 0.5, 0.75, 0.95]:
        sub = df[np.isclose(df.majority_fraction, p)].set_index("solution")
        winners[f"p = {p:.2f}"] = [" / ".join(argmax_all(sub[m])) for m in SWEEP_METRICS]
    winners = pd.DataFrame(winners, index=pd.Index([label(m) for m in SWEEP_METRICS], name="Metric"))
    return df, cross, winners


# --- Experiment 4: ranking comparison --------------------------------------------------


def candidate_solutions(X, y) -> dict[str, np.ndarray]:
    sizes = np.bincount(y)
    largest = int(sizes.argmax())
    smallest_two = list(np.argsort(sizes)[:2])
    sols = {}
    for k in [2, 3, 4, 5, 6, 8, 10]:
        sols[f"k-means k={k}"] = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
    for linkage in ["ward", "average", "single"]:
        sols[f"agglo-{linkage} k={TRUE_K}"] = AgglomerativeClustering(
            TRUE_K, linkage=linkage
        ).fit_predict(X)
    sols[f"GMM k={TRUE_K}"] = GaussianMixture(TRUE_K, random_state=SEED).fit(X).predict(X)
    sols["truth + 10% noise"] = perturb.label_noise(y, 0.1, rng=SEED)
    sols["truth, split largest"] = perturb.fragment(y, [largest], 2, rng=SEED)
    sols["truth, merge 2 smallest"] = perturb.merge(y, [smallest_two])
    sols["single cluster"] = perturb.single_cluster(y)
    sols["all singletons"] = perturb.singletons(y)
    return sols


def is_algorithmic(solution: str) -> bool:
    return not (solution.startswith("truth") or solution in ("single cluster", "all singletons"))


def analyse_pool(metric_scores: pd.DataFrame) -> dict:
    return {
        "metric_scores": metric_scores,
        "ranks": rank_solutions(metric_scores),
        "tau": kendall_matrix(metric_scores),
        "dis": disagreement_rate(metric_scores),
        "winners": pd.DataFrame(
            {"Top-ranked solution(s)": [", ".join(argmax_all(metric_scores[m])) for m in RANKED]},
            index=pd.Index([label(m) for m in RANKED], name="Metric"),
        ),
    }


def pool_summary(pool: dict) -> dict:
    out = {}
    for group_name, group in [("all 10 metrics", RANKED), ("6 two-sided metrics", TWO_SIDED)]:
        t = pool["tau"].loc[group, group].to_numpy()
        d = pool["dis"].loc[group, group].to_numpy()
        iu = np.triu_indices(len(group), 1)
        out[f"mean Kendall tau ({group_name})"] = np.nanmean(t[iu])
        out[f"min Kendall tau ({group_name})"] = np.nanmin(t[iu])
        out[f"mean disagreement rate ({group_name})"] = np.mean(d[iu])
    ranks = pool["ranks"][TWO_SIDED]
    out["mean rank spread across two-sided metrics"] = float((ranks.max(axis=1) - ranks.min(axis=1)).mean())
    return out


def exp_rankings(data):
    results = {}
    for name, (X, y) in data.items():
        sols = candidate_solutions(X, y)
        scores = pd.DataFrame({s: evaluate(y, p) for s, p in sols.items()}).T
        scores.index.name = "Solution"
        scores.to_csv(OUT / f"exp4_scores_{name}.csv")
        metric_scores = scores[RANKED].astype(float)
        full = analyse_pool(metric_scores)
        algo = analyse_pool(metric_scores.loc[[s for s in metric_scores.index if is_algorithmic(s)]])
        for pool_name, pool in [("all", full), ("algorithms", algo)]:
            pool["ranks"].to_csv(OUT / f"exp4_ranks_{name}_{pool_name}.csv")
            pool["tau"].to_csv(OUT / f"exp4_kendall_{name}_{pool_name}.csv")
            pool["dis"].to_csv(OUT / f"exp4_disagreement_{name}_{pool_name}.csv")
        results[name] = {"scores": scores, "all": full, "algorithms": algo}

    for pool_name in ["all", "algorithms"]:
        fig, axes = plt.subplots(1, 2, figsize=(17, 7.5 if pool_name == "all" else 6.5))
        for ax, name in zip(axes, data):
            heatmap(ax, results[name][pool_name]["ranks"], "{:d}", "viridis_r")
            ax.set_title(f"{name} data: rank per metric (1 = best)")
        fig.tight_layout()
        fig.savefig(FIG / f"exp4_ranks_{pool_name}.png", dpi=130)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(19, 8), layout="constrained")
        for ax, name in zip(axes, data):
            im = heatmap(ax, results[name][pool_name]["tau"], "{:.2f}", "RdBu", -1, 1)
            ax.set_title(f"{name} data: Kendall tau between metric rankings")
        fig.colorbar(im, ax=axes, shrink=0.8)
        fig.savefig(FIG / f"exp4_kendall_{pool_name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    summary = pd.DataFrame(
        {
            f"{name} / {pool_name}": pool_summary(results[name][pool_name])
            for pool_name in ["all", "algorithms"]
            for name in data
        }
    )
    summary.index.name = "Statistic"
    summary.to_csv(OUT / "exp4_summary.csv")
    return results, summary


def top_conflicts(metric_scores, a, b, n=3):
    c = pairwise_conflicts(metric_scores, a, b)
    if c.empty:
        return c
    c["strength"] = np.minimum(c.delta_a.abs(), c.delta_b.abs())
    return c.sort_values("strength", ascending=False).head(n)


# --- Report --------------------------------------------------------------------------


def fmt_scores(df, cols=RANKED):
    out = df[cols].astype(float).copy()
    out.columns = [label(c) for c in cols]
    return out


def write_report(ex0, e1, e2, e3, e4):
    _, best_k = e1
    frag, merge, single = e2
    imb, cross, imb_winners = e3
    res, summary = e4
    imb_res = res["imbalanced"]
    ms = imb_res["algorithms"]["metric_scores"]
    algo_ranks = imb_res["algorithms"]["ranks"]

    conflict_blocks = []
    for a, b in [("accuracy", "nmi"), ("pair_f1", "ari"), ("bcubed_f1", "accuracy"), ("class_f", "ari")]:
        c = top_conflicts(ms, a, b)
        if c.empty:
            conflict_blocks.append(f"- **{label(a)} vs {label(b)}**: no reversals on this pool.")
            continue
        lines = []
        for _, r in c.iterrows():
            x, y = r.preferred_by_a, r.preferred_by_b
            lines.append(
                f"  - {label(a)} prefers *{x}* ({ms.at[x, a]:.3f} vs {ms.at[y, a]:.3f}); "
                f"{label(b)} prefers *{y}* ({ms.at[y, b]:.3f} vs {ms.at[x, b]:.3f})"
            )
        n_all = len(pairwise_conflicts(ms, a, b))
        conflict_blocks.append(
            f"- **{label(a)} vs {label(b)}** reverse the order of {n_all} solution pairs, e.g.\n"
            + "\n".join(lines)
        )

    single_cluster = imb_res["scores"].loc["single cluster"]
    split_largest = imb_res["scores"].loc["truth, split largest"]
    s_bal, s_imb = summary["balanced / algorithms"], summary["imbalanced / algorithms"]
    tau_key, dis_key = "mean Kendall tau (6 two-sided metrics)", "mean disagreement rate (6 two-sided metrics)"
    spread_key = "mean rank spread across two-sided metrics"
    p95 = imb[np.isclose(imb.majority_fraction, 0.95)].set_index("solution")
    k1 = e1[0][e1[0].dataset == "imbalanced"].set_index("k")
    n_algo = len(algo_ranks)

    def rank_of(sol, m):
        return int(algo_ranks.at[sol, m])

    best_k_imb = {m: argmax_all(k1[m]) for m in RANKED}

    def by_solution(pick) -> str:
        groups: dict[str, list[str]] = {}
        for m in TWO_SIDED:
            groups.setdefault(pick(ms[m]), []).append(label(m))
        return "; ".join(f"*{s}* by {', '.join(ms_)}" for s, ms_ in groups.items())

    full_ms = imb_res["all"]["metric_scores"]
    full_top = {s for m in TWO_SIDED for s in argmax_all(full_ms[m])}
    if len(full_top) == 1:
        top = next(iter(full_top))
        full_pool_lead = (
            f"One controlled corruption, *{top}*, is closer to the truth than any algorithm "
            "output, and every two-sided metric ranks it first. The other corruptions do not "
            "dominate: *truth + 10% noise*, for example, loses to the best algorithms on "
            "every two-sided metric."
        )
    else:
        full_pool_lead = (
            "The two-sided metrics do not even agree on the best solution in the full pool."
        )
    biggest = algo_ranks[TWO_SIDED].max(axis=1) - algo_ranks[TWO_SIDED].min(axis=1)
    widest = biggest.idxmax()
    widest_lo = algo_ranks.loc[widest, TWO_SIDED].idxmin()
    widest_hi = algo_ranks.loc[widest, TWO_SIDED].idxmax()
    merge_bal = single.loc["balanced: merge two classes"]
    merge_mix = single.loc["imbalanced: merge majority + smallest"]
    best_algo = by_solution(lambda s: s.idxmax())
    worst_algo = by_solution(lambda s: s.idxmin())
    p95_single, p95_split = p95.loc["single cluster"], p95.loc["split majority in 2"]
    singletons_nmi = imb_res["scores"].loc["all singletons", "nmi"]
    crossover = cross.iloc[:, 0]
    plateau = k1.loc[TRUE_K - 1 :]
    pur_lo, pur_hi = plateau["purity"].min(), plateau["purity"].max()
    pp_lo, pp_hi = plateau["pair_precision"].min(), plateau["pair_precision"].max()
    p50 = imb[np.isclose(imb.majority_fraction, 0.5)].set_index("solution")["accuracy"]
    acc_winner = imb.loc[imb.groupby("majority_fraction")["accuracy"].idxmax()]
    noise_ps = acc_winner.loc[acc_winner.solution == "20% random noise", "majority_fraction"]

    report = f"""# When clustering metrics disagree

Research question: **how can evaluation metrics produce conflicting conclusions about
clustering quality?**

## Abstract

External clustering metrics compress the contingency matrix between predicted clusters and
true classes into a single number, and each compression encodes a different notion of
error. We implement purity, inverse purity, maximum-matching accuracy, pair-counting
precision, recall and F1, the class F-measure, BCubed, ARI and NMI from one contingency
matrix. We then evaluate them under controlled fragmentation and merging, a class-imbalance
sweep, k-means with k = 1…15, and a ranking of 16 candidate solutions on balanced and
imbalanced data.

One-sided metrics are maximised by degenerate clusterings. Pair recall and inverse purity
peak at k = 1. Purity and pair precision plateau near 0.95 for every k ≥ {TRUE_K - 1} and
reach 1.0 only on refinements of the truth, such as putting every point in its own cluster.
Metrics without chance correction prefer a single all-inclusive cluster over a solution
with one fragmentation error once the majority class is large enough: above exactly
p = 2/3 for accuracy, and somewhere in {crossover[label("bcubed_f1")]} for BCubed F1. ARI,
which is chance-corrected, never does. NMI never does either, but only because a single
cluster carries zero mutual information. NMI is not chance-corrected: it gives "all
singletons" {singletons_nmi:.3f} where ARI gives 0. Among {n_algo} solutions produced by clustering algorithms, the mean Kendall rank
correlation between six two-sided metrics drops from {s_bal[tau_key]:.2f} on balanced data
to {s_imb[tau_key]:.2f} on imbalanced data.

Conflicts arise from four design choices hidden inside each metric: which error type it
penalises, whether it counts points or pairs, how it matches clusters to classes, and
whether it corrects for chance. Class imbalance and a wrong number of clusters amplify all
four.

## Introduction and method

Clustering is unsupervised, so when ground-truth labels exist the natural evaluation is to
compare the predicted partition with the true one. Cluster identifiers are arbitrary,
however, so every external metric first builds the **contingency matrix** (true classes ×
predicted clusters) and then summarises it. The research question is how the choice of
summary can change the verdict.

Two failure modes are central:

- **Fragmentation:** a single true class is split across several clusters.
- **Merging:** several true classes share one cluster.

**Class imbalance** matters because it changes how much each class contributes to a
point-weighted or pair-weighted score.

**Data.** The data are two-dimensional Gaussian blobs with 5 classes around fixed centres
(standard deviation {CLUSTER_STD}, so neighbouring classes overlap). The balanced version
has 5 × 240 points; the imbalanced one has class sizes {DATASETS["imbalanced"]}. The
controlled experiments use synthetic label vectors so that the type and size of every
error is known exactly.

**Clusterings.** k-means (k = 1…15), agglomerative clustering (ward, average and single
linkage), a Gaussian mixture model, and constructed corruptions of the truth (fragmenting a
class, merging classes, random label noise, one cluster, all singletons).

**Comparison of rankings.** For each metric, the solutions are ranked from best to worst.
Agreement between two metrics is measured by Kendall's tau over solution scores and by
the *disagreement rate*: the fraction of solution pairs, among those not tied under
either metric, that the two metrics order in opposite directions.

The implementation is verified by unit tests against textbook values and scikit-learn.
Every table and figure is regenerated by `python experiments/run_all.py`.

## 1. Metrics, all derived from the contingency matrix

Let `n_ij` be the number of points of true class *i* in predicted cluster *j*, `a_i` the
class sizes, `b_j` the cluster sizes and `N` the number of points.

| Metric | Definition | Error it ignores |
|---|---|---|
| Purity | Σ_j max_i n_ij / N | fragmentation (all singletons = 1) |
| Inverse purity | Σ_i max_j n_ij / N | merging (one cluster = 1) |
| Max-matching accuracy | best one-to-one class↔cluster assignment (Hungarian), matched points / N | none, but surplus clusters count fully as errors |
| Pair precision | pairs together in both / pairs together in prediction | fragmentation |
| Pair recall | pairs together in both / pairs together in truth | merging |
| Pair F1 | harmonic mean of pair precision and recall | weights classes by size² |
| Class F-measure | Σ_i (a_i/N) max_j F(i, j) (Larsen & Aone) | one cluster may serve several classes |
| BCubed F1 | per-point precision / recall, then F1 | — |
| ARI | Rand index corrected for chance | — |
| NMI | mutual information / mean entropy | not chance-corrected: lenient towards over-splitting (all singletons > 0); AMI is the corrected form |

### Worked contingency matrix (imbalanced data, k-means with k = 8)

{md_table(ex0["frame"], fmt="{}")}

Purity credits each cluster with its majority class, so it is
{ex0["scores"]["purity"]:.3f}. Max-matching accuracy may use each class only once, so
the three surplus clusters count as errors and accuracy drops to {ex0["acc"]:.3f}. The
Hungarian mapping (cluster → class) is `{ex0["mapping"]}`.

![datasets](figures/datasets.png)

## 2. Experiment 1: varying the number of clusters

K-means with k = 1…15 on 5-class blobs (balanced: 5 × 240 points; imbalanced:
{DATASETS["imbalanced"]}).

{md_table(best_k)}

![vary k](figures/exp1_vary_k.png)

The one-sided metrics are pulled towards the extremes. Inverse purity and pair recall are
maximised by a single cluster (k = 1). Purity and pair precision move the other way, but
on the imbalanced data they flatten out quickly. For every k ≥ {TRUE_K - 1}, purity stays
between {pur_lo:.3f} and {pur_hi:.3f} and pair precision between {pp_lo:.3f} and
{pp_hi:.3f}. Their nominal best k values in the table, k =
{", ".join(map(str, best_k_imb["purity"]))} and k =
{", ".join(map(str, best_k_imb["pair_precision"]))}, are not meaningful peaks within that
plateau. Neither metric penalises extra clusters, and both reach 1.0 only on refinements of
the truth such as all singletons.

On the imbalanced data, every two-sided metric prefers
k = {", ".join(map(str, best_k_imb["accuracy"]))} to the true k = {TRUE_K}. The contingency
matrices show why. At k = {TRUE_K}, k-means spends a centroid splitting the 700-point class
in half, and the 50-point central class still gets no cluster of its own:

{md_table(ex0["frame_k5"], fmt="{}")}

At k = {TRUE_K - 1}, the majority class stays intact and only the tiny class is lost:

{md_table(ex0["frame_k4"], fmt="{}")}

Purity barely notices the split ({k1.at[TRUE_K, "purity"]:.3f} at k = {TRUE_K} vs
{k1.at[TRUE_K - 1, "purity"]:.3f} at k = {TRUE_K - 1}). Max-matching accuracy collapses
from {k1.at[TRUE_K - 1, "accuracy"]:.3f} to {k1.at[TRUE_K, "accuracy"]:.3f} because one
half of the majority class can no longer be matched. This is the same data and the same
pair of clusterings, but purity and accuracy reach opposite conclusions.

## 3. Experiment 2: fragmentation vs merging

Ground truth of 8 classes × 100 points; either split every class into *m* clusters or
merge *g* classes into one.

![fragmentation and merging](figures/exp2_frag_merge.png)

{md_table(fmt_scores(frag.set_index("pieces")))}

*Fragmentation (rows = pieces per class).*

{md_table(fmt_scores(merge.set_index("merged")))}

*Merging (rows = number of classes merged).*

Fragmentation is invisible to purity, pair precision and BCubed precision, all of which
stay at 1.0. Merging is invisible to inverse purity, pair recall and BCubed recall. So any
single one-sided score can rank a badly fragmented solution above a mildly merged one, or
the reverse.

### One split vs one merge

{md_table(fmt_scores(single))}

*Separate synthetic labellings, distinct from the blob data and from Experiment 3. Balanced
rows: {len(SINGLE_ERROR_BALANCED)} classes × {SINGLE_ERROR_BALANCED[0]} points. Imbalanced
rows: class sizes {SINGLE_ERROR_IMBALANCED} ({sum(SINGLE_ERROR_IMBALANCED)} points).*

The same structural mistake costs very different amounts depending on which class it
hits. On imbalanced data, splitting the majority class in half costs pair F1 a lot because
it destroys a quadratic number of pairs. NMI, by contrast, stays relatively high
({single.at["imbalanced: split majority in 2", "nmi"]:.3f} against an ARI of
{single.at["imbalanced: split majority in 2", "ari"]:.3f}). Merging the two smallest classes
barely registers under pair-based metrics (pair F1
{single.at["imbalanced: merge two smallest", "pair_f1"]:.3f}), but max-matching accuracy
charges for every point of the unmatched class
({single.at["imbalanced: merge two smallest", "accuracy"]:.3f}).

## 4. Experiment 3: class imbalance

Four classes and 1000 points; the majority class holds a fraction *p*, and the rest is
split evenly over three minority classes.

![imbalance](figures/exp3_imbalance.png)

**When does a degenerate one-cluster answer beat a solution with one fragmentation error
(perfect except that the majority class is split in two)?**

{md_table(cross, fmt="{}")}

The sweep moves *p* in steps of 0.05, so each crossover is only known to lie within one
step. Accuracy has a closed form: the single cluster scores *p* and the split scores
1 − *p*/2, so they cross at exactly *p* = 2/3.

**Best solution per metric as imbalance grows:**

{md_table(imb_winners, fmt="{}")}

The p = 0.50 row looks surprising: accuracy prefers 20% random noise
({p50["20% random noise"]:.3f}) to both structural errors (split majority
{p50["split majority in 2"]:.3f}, merge minorities {p50["merge all minorities"]:.3f}). Random
noise costs about 20% of the points whatever the class sizes. Splitting the majority costs
*p*/2 of the points, and merging the three minorities costs 2(1 − *p*)/3. Noise therefore
wins under accuracy whenever both structural errors cost more than 20%, which in this sweep
happens for p = {", ".join(f"{p:.2f}" for p in noise_ps)}.

At p = 0.95 the single cluster scores purity {p95.at["single cluster", "purity"]:.3f},
accuracy {p95.at["single cluster", "accuracy"]:.3f} and pair F1
{p95.at["single cluster", "pair_f1"]:.3f}, but ARI {p95.at["single cluster", "ari"]:.3f}
and NMI {p95.at["single cluster", "nmi"]:.3f}. The two zeros have different causes. ARI is
chance-corrected, so any labelling that is no better than random scores 0. NMI is not
chance-corrected: it is 0 here only because a single cluster shares no information with
the classes, and it still rewards over-splitting (all singletons score
{singletons_nmi:.3f} on the imbalanced blobs). Metrics without chance correction inherit
the majority-class baseline, so their absolute values become hard to interpret under
imbalance, and their rankings can flip in favour of trivial solutions.

## 5. Experiment 4: do metrics rank the same solutions the same way?

Each dataset gets 16 candidate solutions. {n_algo} are produced by algorithms: k-means for
several k, agglomerative clustering (ward, average and single linkage) and a GMM. The other
5 are controlled corruptions of the truth (10% noise, split largest class, merge two
smallest classes) plus the two degenerate answers (one cluster, all singletons). Rankings
are analysed for the full pool and for the {n_algo} algorithmic solutions alone, which is
the realistic model-selection setting.

### Full pool (16 solutions)

![ranks, full pool](figures/exp4_ranks_all.png)

{full_pool_lead}

{md_table(imb_res["all"]["winners"], fmt="{}")}

*(imbalanced data; ties share rank 1)*

Even here, the one-sided metrics crown a degenerate answer. Purity and pair precision
give their maximum of 1.0 to "all singletons" as well as to the fragmented truth, while
inverse purity and pair recall give theirs to "single cluster" as well as to the merged
truth.

### Algorithmic solutions only ({n_algo} solutions)

![ranks, algorithms](figures/exp4_ranks_algorithms.png)

![kendall, algorithms](figures/exp4_kendall_algorithms.png)

{md_table(summary)}

*"Rank spread" is the average, over solutions, of (worst rank − best rank) that the six
two-sided metrics give the same solution.*

On the imbalanced data, the best algorithm according to the two-sided metrics is
{best_algo}. They do not agree on the worst one: {worst_algo}. The disagreements are not
limited to near-ties either. The widest gap among the two-sided metrics is for
*{widest}*, which is ranked {int(algo_ranks.at[widest, widest_lo])} of {n_algo} by
{label(widest_lo)} but {int(algo_ranks.at[widest, widest_hi])} by {label(widest_hi)}.
Including one-sided metrics widens it further: over-split k-means with k = 10 is
ranked {rank_of("k-means k=10", "purity")} by purity and
{rank_of("k-means k=10", "accuracy")} by accuracy.

### Concrete reversals (imbalanced data, algorithmic solutions)

{chr(10).join(conflict_blocks)}

### Full scores (imbalanced data)

{md_table(fmt_scores(imb_res["scores"]))}

## 6. Answer: how metrics come to conflicting conclusions

1. **They measure different error types.** Purity, pair precision and BCubed precision
   penalise only merging, and their mirror images penalise only fragmentation. Any
   one-sided score can be maximised by a degenerate clustering (all singletons or a single
   cluster), so it orders solutions by their number of clusters as much as by their
   quality.
2. **They count different units.** Point-based metrics (purity, accuracy, BCubed) weight a
   class linearly by its size. Pair-based metrics (pair P/R/F, ARI) weight it
   quadratically. Errors inside the majority class therefore dominate pair metrics, and
   errors in small classes almost vanish from them.
3. **They match clusters to classes differently.** Purity lets several clusters claim the
   same class, the class F-measure lets several classes claim the same cluster, and
   max-matching accuracy allows only a one-to-one mapping. A fragmented solution is free
   under purity but pays in full under accuracy. For merges the outcome depends on the
   class sizes involved. Merging two equal classes is penalised less by the class
   F-measure ({merge_bal["class_f"]:.3f}) than by accuracy ({merge_bal["accuracy"]:.3f}),
   because one cluster can be the best match for both classes. Merging the majority with
   the smallest class reverses this (class F {merge_mix["class_f"]:.3f}, accuracy
   {merge_mix["accuracy"]:.3f}): the small class loses almost all of its precision, while
   accuracy still credits every majority point.
4. **Only some of them correct for chance.** With a 95% majority class, putting everything
   in one cluster scores accuracy {p95_single["accuracy"]:.3f} and pair F1
   {p95_single["pair_f1"]:.3f}, while ARI = {p95_single["ari"]:.3f}. A solution with one
   fragmentation error (the majority split in two) scores accuracy
   {p95_split["accuracy"]:.3f} and pair F1 {p95_split["pair_f1"]:.3f}, but ARI
   {p95_split["ari"]:.3f}. Uncorrected metrics inherit the majority-class baseline, so they
   rank the useless answer higher once imbalance passes a metric-specific threshold:
   exactly p = 2/3 for accuracy, and somewhere in {crossover[label("pair_f1")]} for pair F1
   (see Experiment 3). Of the metrics studied, only ARI is chance-corrected. NMI also gives
   the single cluster 0, but only because it shares no information with the classes; NMI
   still gives all singletons {singletons_nmi:.3f}. The chance-corrected form of NMI is the
   adjusted mutual information, AMI (Vinh et al., 2010). On the moderately imbalanced blobs
   (58% majority) the comparison does not flip: the single cluster's accuracy of
   {single_cluster["accuracy"]:.3f} is still below {split_largest["accuracy"]:.3f}.

**What amplifies them**

- **A wrong number of clusters.** One-sided metrics reward moving k in one direction:
  purity and pair precision plateau as k grows, while inverse purity and pair recall peak
  at k = 1. Two-sided metrics differ in how hard they punish a wrong k. NMI is lenient
  towards over-splitting and harsh on under-splitting: on the imbalanced data it ranks
  k-means with k = 2 at {rank_of("k-means k=2", "nmi")} of {n_algo} (accuracy: rank
  {rank_of("k-means k=2", "accuracy")}) and k-means with k = 10 at
  {rank_of("k-means k=10", "nmi")} of {n_algo} (accuracy: rank
  {rank_of("k-means k=10", "accuracy")}). Accuracy charges every surplus cluster in full.
- **Class imbalance.** Among the algorithmic solutions, the mean Kendall tau between the
  six two-sided metrics falls from {s_bal[tau_key]:.2f} on balanced data to
  {s_imb[tau_key]:.2f} on imbalanced data. A typical pair of these metrics orders
  {100 * s_bal[dis_key]:.0f}% of solution pairs oppositely on balanced data and
  {100 * s_imb[dis_key]:.0f}% on imbalanced data, and the average rank spread of a solution
  grows from {s_bal[spread_key]:.1f} to {s_imb[spread_key]:.1f} places.

**Practical consequence:** a single number cannot certify a clustering as better. Report
a complementary pair that exposes both error types, such as pair or BCubed precision and
recall, or purity and inverse purity. Add a chance-corrected score (ARI, or AMI rather than
NMI) and show the contingency matrix itself, which is the only view that records exactly
which classes were split or merged.

## References

- Amigó, E., Gonzalo, J., Artiles, J. & Verdejo, F. (2009). A comparison of extrinsic
  clustering evaluation metrics based on formal constraints. *Information Retrieval*,
  12(4), 461–486.
- Bagga, A. & Baldwin, B. (1998). Entity-based cross-document coreferencing using the
  vector space model. *Proc. COLING-ACL*, 79–85.
- Hubert, L. & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2,
  193–218.
- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research
  Logistics Quarterly*, 2, 83–97.
- Larsen, B. & Aone, C. (1999). Fast and effective text mining using linear-time document
  clustering. *Proc. ACM SIGKDD*, 16–22.
- Manning, C. D., Raghavan, P. & Schütze, H. (2008). *Introduction to Information
  Retrieval*, ch. 16. Cambridge University Press.
- Strehl, A. & Ghosh, J. (2002). Cluster ensembles — a knowledge reuse framework for
  combining multiple partitions. *Journal of Machine Learning Research*, 3, 583–617.
- Vinh, N. X., Epps, J. & Bailey, J. (2010). Information theoretic measures for clusterings
  comparison. *Journal of Machine Learning Research*, 11, 2837–2854.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = {name: make_dataset(sizes) for name, sizes in DATASETS.items()}
    ex0 = exp_datasets(data)
    e1 = exp_vary_k(data)
    e2 = exp_frag_merge()
    e3 = exp_imbalance()
    e4 = exp_rankings(data)
    write_report(ex0, e1, e2, e3, e4)
    print(e1[1].to_string())
    print(e3[1].to_string())
    print(e4[1].to_string())
    print(e4[0]["imbalanced"]["algorithms"]["winners"].to_string())
    print(f"Report written to {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
