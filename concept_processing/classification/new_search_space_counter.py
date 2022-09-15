"""
Experimental file attempting to improve upon the approximation for logic-based learning prior.
Not complete. It currently handles the case when we only can have at most one predicate and not negation
"""

from typing import List, Dict, Any, Iterator

import numpy as np


class Predicate:
    def __init__(self, types: np.ndarray):
        self.types = types


# def out_var_iterator(default_dict: Dict[str, int]) -> Iterator[Dict[str, int]]:
#     start_dict = {k: 1 for k in default_dict}

def output_var_iterator(d: Dict[str, int]) -> Iterator[Dict[str, int]]:
    if d == {}:
        # Special case to allow a for loop iteration when output variables do not exist
        yield {}
        return

    keys, values = list(zip(*d.items()))
    upper_bound = list(values)
    curr = [1] * len(values)

    while curr <= upper_bound:
        yield {k: v for k, v in zip(keys, curr)}
        curr[-1] += 1
        for i in range(len(curr) - 1, 0, -1):
            if curr[i] > upper_bound[i]:
                curr[i - 1] += 1
                curr[i] = 1


def solve(predicates: List[Predicate], assigned_vars: Dict[str, int], unassigned_var_cnt: int) -> Dict[int, int]:
    if not predicates:
        # 1 rule of length 1 can exist (just head)
        return {1: 1}

    rule_cnt_dict = {1: 1}
    # TODO: repeated predicates possible, not as well
    for i, predicate in enumerate(predicates):
        tys = predicate.types

        # for 1
        # n_assigned_vars * solve(predicates[1:], assigned_vars, unassigned_var_cnt)
        # + solve(predicates[1:, assigned_vars ++ [type], unassigned_var_cnt - 1)

        # (n_assigned_vars(type) C needed_vars(type)) * solve(predicates[1:, assinged_vars, unassigned_var_cnt)
        #

        for j in range(0, 1 << len(tys)):
            bin_mask = to_bin_mask(j, length=len(tys))
            output_vars = tys[bin_mask == 0]
            out_var_cnt_dict = to_cnt_dict(output_vars)
            for c_out_var_cnt_dict in output_var_iterator(out_var_cnt_dict):
                input_vars = tys[bin_mask]

                new_rule_cnts = count_possibilities(assigned_vars, input_vars, predicates[i + 1:], c_out_var_cnt_dict,
                                                    unassigned_var_cnt)
                rule_cnt_dict = merge_dicts(rule_cnt_dict, new_rule_cnts)

    return rule_cnt_dict


def count_possibilities(assigned_vars: Dict[str, int], input_vars: np.ndarray, remaining_predicates: List[Predicate],
                        out_var_cnt_dict: Dict[str, int], unassigned_var_cnt: int):

    new_assigned_vars = merge_dicts(assigned_vars, out_var_cnt_dict)
    used_output_vars = sum(out_var_cnt_dict.values())

    if used_output_vars <= unassigned_var_cnt:
        possibilities = 1
        for ty, cnt in to_cnt_dict(input_vars).items():
            if ty not in assigned_vars:
                possibilities = 0
            else:
                assigned_cnt = assigned_vars.get(ty, 0)
                # Can choose one of assigned_cnt options for each position
                possibilities *= assigned_cnt ** cnt

        if possibilities > 0:
            rule_cnts = solve(remaining_predicates, new_assigned_vars, unassigned_var_cnt - used_output_vars)
            new_rule_cnts = {k + 1: v * possibilities for k, v in rule_cnts.items()}
            return new_rule_cnts
    return {}


def merge_dicts(dict1: Dict[Any, int], dict2: Dict[Any, int]):
    available_keys = set(dict1.keys()).union(dict2.keys())

    return {key: dict1.get(key, 0) + dict2.get(key, 0) for key in available_keys}


# Expects 1D string np array
def to_cnt_dict(arr: np.ndarray):
    sol = {}
    for x in arr:
        curr_cnt = sol.get(x, 0)
        sol[x] = curr_cnt + 1
    return sol


def to_bin_mask(x: int, length: int):
    output_vars_bin = bin(x)
    arr = [int(d) for d in str(output_vars_bin)[2:]]
    prepend = [0] * (length - len(arr))
    bin_mask = np.array(prepend + arr)
    return bin_mask.astype(bool)
    # print(tys[bin_mask])
