import unittest

import numpy as np
from sklearn import metrics as skm

from clustermetrics import (
    adjusted_rand_index,
    bcubed,
    class_f_measure,
    contingency_matrix,
    evaluate,
    inverse_purity,
    max_matching_accuracy,
    normalized_mutual_info,
    pair_counts,
    pair_precision_recall_f,
    purity,
)
from clustermetrics import perturb


# Manning, Raghavan & Schuetze (IR book, Fig. 16.4): 17 points, 3 clusters.
# Classes: x=0, o=1, d=2.
IR_TRUE = np.array([0] * 5 + [1] * 1 + [0] * 1 + [1] * 4 + [2] * 1 + [0] * 2 + [2] * 3)
IR_PRED = np.array([0] * 6 + [1] * 6 + [2] * 5)


class TestContingency(unittest.TestCase):
    def test_matrix_matches_sklearn(self):
        rng = np.random.default_rng(0)
        y, p = rng.integers(0, 4, 300), rng.integers(0, 6, 300)
        np.testing.assert_array_equal(
            contingency_matrix(y, p).matrix, skm.cluster.contingency_matrix(y, p)
        )

    def test_arbitrary_label_values(self):
        c = contingency_matrix(["a", "a", "b"], [10, 20, 20])
        np.testing.assert_array_equal(c.matrix, [[1, 1], [0, 1]])
        self.assertEqual(list(c.classes), ["a", "b"])


class TestTextbookValues(unittest.TestCase):
    def test_purity(self):
        self.assertAlmostEqual(purity(IR_TRUE, IR_PRED), 12 / 17)

    def test_pair_counts(self):
        pc = pair_counts(IR_TRUE, IR_PRED)
        self.assertEqual((pc["tp"], pc["fp"], pc["fn"], pc["tn"]), (20, 20, 24, 72))

    def test_pair_prf(self):
        p, r, f = pair_precision_recall_f(IR_TRUE, IR_PRED)
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(r, 20 / 44)
        self.assertAlmostEqual(f, 2 * 0.5 * (20 / 44) / (0.5 + 20 / 44))


class TestAgainstSklearn(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.cases = [
            (rng.integers(0, k1, n), rng.integers(0, k2, n))
            for n, k1, k2 in [(50, 2, 2), (200, 3, 7), (500, 6, 3), (120, 5, 5)]
        ]
        self.cases.append((IR_TRUE, IR_PRED))

    def test_ari(self):
        for y, p in self.cases:
            self.assertAlmostEqual(adjusted_rand_index(y, p), skm.adjusted_rand_score(y, p))

    def test_nmi(self):
        for y, p in self.cases:
            self.assertAlmostEqual(
                normalized_mutual_info(y, p), skm.normalized_mutual_info_score(y, p)
            )

    def test_pair_precision_recall_via_fowlkes_mallows(self):
        for y, p in self.cases:
            prec, rec, _ = pair_precision_recall_f(y, p)
            self.assertAlmostEqual(np.sqrt(prec * rec), skm.fowlkes_mallows_score(y, p))

    def test_rand_index(self):
        for y, p in self.cases:
            pc = pair_counts(y, p)
            ri = (pc["tp"] + pc["tn"]) / sum(pc.values())
            self.assertAlmostEqual(ri, skm.rand_score(y, p))


class TestProperties(unittest.TestCase):
    def setUp(self):
        self.y = np.repeat(np.arange(4), [40, 30, 20, 10])

    def test_perfect_labelling(self):
        perm = np.array([3, 0, 2, 1])[self.y]
        for name, value in evaluate(self.y, perm).items():
            if name != "n_clusters":
                self.assertAlmostEqual(value, 1.0, msg=name)

    def test_accuracy_matches_permutation(self):
        p = perturb.label_noise(self.y, 0.2, rng=1)
        acc, mapping = max_matching_accuracy(self.y, p)
        mapped = np.array([mapping[v] for v in p])
        self.assertAlmostEqual(acc, (mapped == self.y).mean())

    def test_accuracy_penalises_extra_clusters(self):
        p = perturb.fragment(self.y, [0], 2, rng=0)
        acc, _ = max_matching_accuracy(self.y, p)
        self.assertAlmostEqual(acc, 80 / 100)

    def test_fragmentation_keeps_purity_and_precision_perfect(self):
        p = perturb.fragment_all(self.y, 3, rng=0)
        self.assertEqual(purity(self.y, p), 1.0)
        self.assertEqual(pair_precision_recall_f(self.y, p)[0], 1.0)
        self.assertEqual(bcubed(self.y, p)[0], 1.0)
        self.assertLess(inverse_purity(self.y, p), 1.0)

    def test_merging_keeps_inverse_purity_and_recall_perfect(self):
        p = perturb.merge(self.y, [[0, 1]])
        self.assertEqual(inverse_purity(self.y, p), 1.0)
        self.assertEqual(pair_precision_recall_f(self.y, p)[1], 1.0)
        self.assertEqual(bcubed(self.y, p)[1], 1.0)
        self.assertLess(purity(self.y, p), 1.0)

    def test_degenerate_solutions(self):
        one = evaluate(self.y, perturb.single_cluster(self.y))
        self.assertAlmostEqual(one["purity"], 0.4)
        self.assertAlmostEqual(one["accuracy"], 0.4)
        self.assertAlmostEqual(one["pair_recall"], 1.0)
        self.assertAlmostEqual(one["ari"], 0.0)
        self.assertAlmostEqual(one["nmi"], 0.0)
        sing = evaluate(self.y, perturb.singletons(self.y))
        self.assertAlmostEqual(sing["purity"], 1.0)
        self.assertAlmostEqual(sing["pair_precision"], 1.0)
        self.assertAlmostEqual(sing["pair_recall"], 0.0)
        self.assertAlmostEqual(sing["ari"], 0.0)

    def test_class_f_bounds(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            p = rng.integers(0, rng.integers(1, 8), self.y.size)
            self.assertTrue(0.0 <= class_f_measure(self.y, p) <= 1.0)


if __name__ == "__main__":
    unittest.main()
