import math
import random
from typing import List, Iterator, Union, Tuple

import numpy as np

from concept_processing.asp.asp_solver import clingo_solve


# Gets the context string of a concept bottlenck model result
def get_concept_bottleneck_context(categories: np.ndarray, nn_outcomes: List[np.ndarray], label: str = None,
                                   concepts_text=None) -> str:
    ex_generator = ClassificationExampleGenerator(list(categories))
    ctxs_atoms = ['concept({}).' for i in range(len(nn_outcomes))]
    ex_generator.parse(nn_outcomes, ctxs_atoms, label)
    ctx = ex_generator.get_context_multi_label()

    if concepts_text is not None:
        for i, explanation in enumerate(concepts_text):
            ctx = ctx.replace(f'concept({i})', f'concept("{explanation}")')

    return ctx


# Predict the label using the logic-based learning method
def predict(ctx: str, sol_file_loc: str, background_loc: str) -> str:
    ctx_loc = '/tmp/ctx_file.out'
    with open(ctx_loc, 'w') as f:
        # for example in ex_generator.get_examples(n_of_predicates=n_explanations):
        f.write(ctx)
    out_atoms = clingo_solve(background_loc, ctx_loc, sol_file_loc)
    assert len(out_atoms) == 1
    sol_str = out_atoms[0][0]
    sol_str = sol_str.lstrip('selected').lstrip('(').rstrip(')')
    return sol_str


# Computes the classification examples for the logic-based learning method
class ClassificationExampleGenerator:
    def __init__(self, categories: List[str]):
        self.categories = categories

        # Number of samples
        self.I = 5  # 100
        # Large number to reduce the error of rounding to int
        self.K = 1000
        # The probablity that an example is incorrect
        self.epsilon = 0.1  # 0.0000001

        # The number of rules we expect with length 1
        self.r_1 = 100
        # The multiplier from the number of rules we expect of length n compared to n-1
        self.r_mult = 2 / 3
        self.max_rule_len = 10

        self._nn_outcomes = []
        self._ctx_atoms = []
        self._labels = []

        random.seed(0)

    # If multilabel - receive List[str] as input, otherwise it is multiclass so we receive List[List[str]]
    def _sample_examples(self, probs_arr: Union[List[List[float]], List[str]], label: Union[str, int], ex_num: int,
                         ctx_atoms: List[str], multilabel=True, out_offset=0) -> List[str]:
        assert len(probs_arr) == len(ctx_atoms)

        exclusion_str, inclusion_str = self._get_exc_inc_str(label)

        penalty = np.round(-self.K * np.log(self.epsilon / (1 - self.epsilon)) / self.I).astype(int)
        probs_arr = np.array(probs_arr)

        indices_cnt = {}
        for i in range(self.I):
            indices = tuple(self._get_true_indices(probs_arr, multilabel))
            indices_cnt[indices] = indices_cnt.get(indices, 0) + 1

        examples = []
        for i, (indices, cnt) in enumerate(indices_cnt.items()):
            ctx_str = self._get_ctx_str(indices, ctx_atoms, out_offset)
            ex_id = f"ex_{ex_num}_{i}"
            scaled_penalty = cnt * penalty
            examples.append(self._str_example(ex_id, scaled_penalty, inclusion_str, exclusion_str, ctx_str))

        return examples

    def _get_ctx_str(self, indices: Tuple[int], ctx_atoms: Iterator[str], out_offset=0):
        context_literals = [ctx_atom.format(idx + out_offset) for idx, ctx_atom in zip(indices, ctx_atoms)]
        ctx_str = '\n'.join(context_literals)
        return ctx_str

    def _get_exc_inc_str(self, label: Union[str, int]):
        if type(label) == int:
            label = self.categories[label]
        make_selected = lambda l: f"selected({l})"

        inclusion_str = make_selected(label)
        exclusion_str = ','.join(map(make_selected, set(self.categories) - {label}))
        return exclusion_str, inclusion_str
        #
        # if label == "invalid":
        #     return "", make_selected("invalid")
        # return make_selected("invalid"), ""

    # number of predicates that ILASP would generate (excluding type matching)
    def _bias_as_str(self, n_of_predicates: int) -> str:
        n_of_classes = len(self.categories)

        rule_len = min(self.max_rule_len, n_of_predicates)

        ns = np.array([(n_of_classes - 1) * (math.comb(n_of_predicates, i)) for i in range(1, rule_len)])
        rs = np.array([self.r_1 * (self.r_mult ** (i - 1)) for i in range(1, rule_len)])

        # Additonal customisable factor for the reduction of the prior value
        prior_scale_factor = 1 / 20
        pens = np.round(prior_scale_factor * self.K * np.log((ns - rs) / rs)).astype(int)

        pen_strs = [f"pen({i + 1}, {pen})" for i, pen in enumerate(pens)]
        str_pens = "\n".join([f'#bias("{pen_str}.").' for pen_str in pen_strs])
        return f'''
#bias("penalty(P, custom) :- L = #count{{X : in_head(X); X : in_body(X)}}, pen(L, P).").
{str_pens}
#bias("pen(L, 100000000000) :- L = #count{{X : in_head(X); X : in_body(X)}}, L >= {rule_len}.").
 '''

    # nn_outcome - list probabilities returned returned by the final layer of a neural network, e.g softmax
    #            - it's length must match that of ctx_atoms as sampling is applied pairwise
    # label - label string or label id in categories
    def parse(self, nn_outcomes: List[np.ndarray], ctxs_atoms: List[str], label: Union[str, int]):
        assert len(nn_outcomes) == len(ctxs_atoms)

        self._nn_outcomes.append(nn_outcomes)
        self._ctx_atoms.append(ctxs_atoms)
        self._labels.append(label)

    # out_offset - whether the found numbers are offset in a logical example, e.g. we may want to count sudoku numbers
    # from 1 instead of 0
    def get_examples(self, n_of_predicates: int, multilabel: bool = True, out_offset=0) -> \
            Iterator[str]:
        for i, (concept_outcome, label, ctx_atoms) in enumerate(zip(self._nn_outcomes, self._labels, self._ctx_atoms)):
            for example in self._sample_examples(concept_outcome, label, i, ctx_atoms, multilabel, out_offset):
                yield example
        yield self._bias_as_str(n_of_predicates)

    def get_context(self, out_offset=0) -> str:
        assert len(self._nn_outcomes) == 1

        concept_outcomes = self._nn_outcomes[0]
        context_atoms = self._ctx_atoms[0]
        indices = np.argmax(concept_outcomes, axis=1)
        return self._get_ctx_str(indices, context_atoms, out_offset)

    def get_context_multi_label(self, out_offset=0) -> str:
        assert len(self._nn_outcomes) == 1

        concept_outcomes = self._nn_outcomes[0]
        context_atoms = self._ctx_atoms[0]
        indices, = np.nonzero(concept_outcomes >= 0.5)
        # indices = np.argmax(concept_outcomes, axis=1)
        return self._get_ctx_str(indices, context_atoms, out_offset)

    # concept_outcomes is a binary vector in this case. Has the same behaviour as get_prob_examples if the model
    # predicted with complete confidence the true label
    def get_no_prob_examples(self, concept_outcomes: np.ndarray, labels: List[str], ids: List[str],
                             ctxs_atoms: Iterator[Iterator[str]], penalty=1) -> Iterator[str]:
        for i, (concept_outcome, label, id_, ctx_atoms) in enumerate(zip(concept_outcomes, labels, ids, ctxs_atoms)):
            exclusion_str, inclusion_str = self._get_exc_inc_str(label)
            true_indices, = np.nonzero(concept_outcome)
            ctx_str = self._get_ctx_str(true_indices, ctx_atoms)
            ex_id = f"ex_{i}_{id_}"
            yield self._str_example(ex_id, penalty, inclusion_str, exclusion_str, ctx_str)

        yield '''
#bias("penalty(1, head(X)) :- in_head(X).").
#bias("penalty(1, body(X)) :- in_body(X).").
'''

    @staticmethod
    def _get_true_indices(probs: np.ndarray, multilabel: bool) -> np.ndarray:
        if multilabel:
            return ClassificationExampleGenerator._get_binary_indices(probs)

        sol = []
        for softmax_vec in probs:
            sol.append(ClassificationExampleGenerator._get_index_in_softmax_vec(softmax_vec))
        return np.array(sol)

    @staticmethod
    def _get_binary_indices(probs: np.ndarray) -> np.ndarray:
        sample = np.random.sample(probs.shape)
        one_hot = (probs >= sample)
        indices, = np.nonzero(one_hot)
        return indices

    @staticmethod
    def _get_index_in_softmax_vec(softmax_vec: np.ndarray) -> int:
        val = np.random.sample()
        for i, prob in enumerate(softmax_vec):
            if prob > val:
                return i
            val -= prob
        return -1

    @staticmethod
    def _str_example(ex_id: str, penalty: int, inclusion_str: str, exclusion_str: str, context_str: str,
                     is_positive: bool = True) -> str:
        prefix = "pos" if is_positive else "neg"
        return f"#{prefix}({ex_id}@{penalty},\n" \
               f"{{ {inclusion_str} }},\n" \
               f"{{ {exclusion_str} }}, \n" \
               f"{{\n" \
               f"{context_str}\n" \
               "}).\n"
