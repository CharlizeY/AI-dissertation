import unittest

import numpy as np

from concept_processing.classification.new_search_space_counter import to_bin_mask, to_cnt_dict, merge_dicts, Predicate, \
    solve, output_var_iterator


class TestMetrics(unittest.TestCase):
    def test_to_bin_mask_works_correctly(self):
        expect = np.array([1, 0])
        x = 2
        self.assertTrue((to_bin_mask(x, length=2) == expect).all())

    def test_to_bin_mask_prepend_works(self):
        expect = np.array([0, 1])
        x = 1
        self.assertTrue((to_bin_mask(x, length=2) == expect).all())

    def test_to_cnt_dictinary(self):
        given = np.array(["hello", "world", "hello", "hi"])
        expect = {"hello": 2, "hi": 1, "world": 1}
        got = to_cnt_dict(given)
        self.assertEqual(got, expect)

    def test_merge_dicts(self):
        dict1 = {"hello": 2, "hi": 1}
        dict2 = {"hello": 2, "world": 1}
        expect = {"hello": 4, "hi": 1, "world": 1}
        got = merge_dicts(dict1, dict2)
        self.assertEqual(got, expect)

    def test_dict_iterator(self):
        expect = [{"a": 1, "b": 1}, {"a": 1, "b": 2}, {"a": 2, "b": 1}, {"a": 2, "b": 2}]
        given = {"a": 2, "b": 2}

        got = list(output_var_iterator(given))

        self.assertEqual(got, expect)

    def test_solve_test_only_pos_vars(self):
        predicate1 = Predicate(types=np.array(['cell', 'row']))
        predicate2 = Predicate(types=np.array(['cell', 'col']))
        predicate3 = Predicate(types=np.array(['cell', 'block']))
        predicate4 = Predicate(types=np.array(['cell', 'cell']))
        predicate5 = Predicate(types=np.array(['cell', 'num']))

        l = [predicate1, predicate2, predicate3, predicate4, predicate5]
        assigned_vars = {}

        rule_cnts = solve(l, assigned_vars, unassigned_var_cnt=4)

        self.assertEqual(rule_cnts[1], 1)
        self.assertEqual(rule_cnts[2], 6)
        self.assertEqual(rule_cnts[3], 32)
        self.assertEqual(rule_cnts[4], 52)



