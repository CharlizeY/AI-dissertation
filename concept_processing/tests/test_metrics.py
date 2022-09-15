import unittest

from concept_processing.metrics import jaccard, precision, recall


class TestMetrics(unittest.TestCase):
    def test_jaccard_0_if_disjoint(self):
        a = [1, 2, 3]
        b = []
        self.assertEqual(jaccard(a, b), 0, "Should be 0")

    def test_jaccard_1_if_equal(self):
        a = [1, 2, 3]
        self.assertEqual(jaccard(a, a), 1, "Should be 1")

    def test_jaccard_usual_case(self):
        a = [1, 2, 3]
        b = [2, 4, 5]
        self.assertEqual(jaccard(a, b), 1 / 5, "Incorrect jaccard computation")

    def test_precision_0_if_disjoint(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        self.assertEqual(precision(a, b), 0, "Should be 0")

    def test_precision_1_if_equal(self):
        a = [1, 2, 3]
        self.assertEqual(precision(a, a), 1, "Should be 1")

    def test_precision_usual_case(self):
        a = [1, 2, 3]
        b = [2, 4, 5]
        self.assertEqual(precision(a, b), 1 / 3, "Incorrect precision computation")

    def test_recall_0_if_disjoint(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        self.assertEqual(recall(a, b), 0, "Should be 0")

    def test_recall_1_if_equal(self):
        a = [1, 2, 3]
        self.assertEqual(recall(a, a), 1, "Should be 1")

    def test_recall_usual_case(self):
        a = [1, 2, 3]
        b = [2, 4, 5]
        self.assertEqual(recall(a, b), 1 / 3, "Incorrect recall computation")
