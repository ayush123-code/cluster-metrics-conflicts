# cluster-metrics-conflicts

How can external clustering metrics reach conflicting conclusions about the same
clusterings? This project builds a contingency matrix between predicted clusters and true
labels and computes the following metrics from it:

- purity and inverse purity
- max-matching (Hungarian) accuracy
- pair-counting precision, recall and F1
- the Larsen & Aone class F-measure
- BCubed precision, recall and F1
- ARI and NMI

It then runs experiments on fragmentation, merging, class imbalance and the number of
clusters, and measures how differently the metrics rank the same solutions.

## Run

```
pip install -r requirements.txt
python -m unittest discover -s tests -t .
python experiments/run_all.py
```

The results go to `results/`: `report.md` (findings plus the answer to the research
question), `figures/*.png` and one CSV per experiment.

## Layout

| Path | Contents |
|---|---|
| `clustermetrics/contingency.py` | contingency matrix and all metrics |
| `clustermetrics/perturb.py` | controlled fragmentation, merging, noise and degenerate labellings |
| `clustermetrics/ranking.py` | per-metric ranks, Kendall tau, pairwise disagreement rate |
| `experiments/run_all.py` | the four experiments and the report generator |
| `tests/test_metrics.py` | checks against textbook values and scikit-learn |
