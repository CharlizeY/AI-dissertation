import itertools
import pickle
import subprocess
from typing import List

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

from concept_processing.asp.asp_solver import clingo_solve
from concept_processing.classification.example_generator import ClassificationExampleGenerator

pam_storage = '/Users/Cherry0904/Desktop/roko-for-charlize/cmd/final_dict_new_codex_no_pruning.pkl'
# pam_storage = '/Users/Cherry0904/Desktop/roko-for-charlize/cmd/final_dict_new_codex.pkl'
output_dir = '../../ilasp/classification/baseball_pam'
output_ex_file = f'{output_dir}/fastlas_examples_pam_no_pruning.txt'
# output_ex_file = f'{output_dir}/fastlas_examples_pam.txt'
output_run_file = f'{output_dir}/solution_fastlas_no_pruning.txt'
# output_run_file = f'{output_dir}/solution_fastlas.txt'

train_test_split = 1700
example_penalty = 1


def get_classification_predictions(concept_list: List[np.ndarray]):
    y_pred = []
    for concepts in concept_list:
        concepts[concepts <= 0.5] = 0
        concepts[concepts > 0.5] = 1
        indices, = np.nonzero(concepts)
        body_literals = [body_atom_fmt.format(idx) for idx in indices]
        tmp_ctx_file = "ctx.tmp"
        with open(tmp_ctx_file, 'w') as f:
            f.write('\n'.join(body_literals))
            # Only care about the final solution
            f.write('\n#show selected/1.\n')

        clingo_out = clingo_solve(f"{output_dir}/background.lp", tmp_ctx_file, output_run_file)
        sol_str = clingo_out[0][0]
        # lstrip selected( has a bug
        sol_str = sol_str.lstrip('selected').lstrip('(').rstrip(')')
        y_pred.append(sol_str)

    return y_pred


if __name__ == '__main__':
    c_dict = pickle.load(open(pam_storage, 'rb'))

    ids = c_dict['id']
    labels = c_dict['label']
    concepts = c_dict['concepts']
    explanations = c_dict['explanations']

    categories = np.unique(labels)

    ex_generator = ClassificationExampleGenerator(list(categories))

    body_atom_fmt = 'concept({}).'
    ctxs_atoms = [itertools.repeat(body_atom_fmt) for _ in range(len(labels))]

    train_concepts = concepts[:train_test_split]
    train_labels = labels[:train_test_split]
    train_ids = ids[:train_test_split]
    train_ctx_atoms = ctxs_atoms[:train_test_split]

    with open(output_ex_file, 'w') as f:
        for i, example in enumerate(
                ex_generator.get_no_prob_examples(train_concepts, train_labels, train_ids, train_ctx_atoms,
                                                  penalty=example_penalty)):
            f.write(example)
            f.write('\n')

    with open(output_run_file, "w") as f:
        subprocess.run([
            "FastLAS", "--nopl",
            f"{output_dir}/background.lp",
            f"{output_dir}/language_bias.ilasp",
            output_ex_file,
            # "--debug"
        ], stdout=f)

    test_concepts = concepts[train_test_split:]
    test_labels = labels[train_test_split:]
    test_ids = ids[train_test_split:]

    y_pred = get_classification_predictions(test_concepts)

    y_true = test_labels

    print(f"Confusion matrix with order {categories}")
    print(confusion_matrix(y_true, y_pred, labels=categories))
    print("Accuracy is:")
    print(accuracy_score(y_true, y_pred))
