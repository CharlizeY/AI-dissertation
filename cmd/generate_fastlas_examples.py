# Dict:
#  - probs np.ndarray - n_training_samples (1700) x n_concepts
#  - explanations - List[str] - len == n_concepts
#  - labels - List[str] - len == n_training_samples
import pickle

import numpy as np

from concept_processing.classification.example_generator import ClassificationExampleGenerator

# conc_vec_storage = '/Users/Cherry0904/Desktop/roko-for-charlize/cmd/for_fastlas_examples_test.pkl'
# conc_vec_storage_tmp = '/Users/Cherry0904/Desktop/roko-for-charlize/cmd/fastlas_examples/for_fastlas_examples_train_{}.pkl'

demo_version = True

conc_vec_storage_tmp = f'/Users/Cherry0904/Desktop/roko-for-charlize/cmd/fastlas_examples/for_fastlas_examples_train{"_attn_" if demo_version else "_"}{{}}.pkl'
ex_file_tmp = f'../ilasp/classification/baseball_model{"_attn" if demo_version else ""}/examples_{{}}.las'


if __name__ == '__main__':

    for i in range(10):
        conc_vec_storage = conc_vec_storage_tmp.format(i)
        out_file = ex_file_tmp.format(i)
        c_dict = pickle.load(open(conc_vec_storage, 'rb'))

        probs = c_dict["probs"]
        explanations = c_dict["explanations"]
        labels = c_dict["labels"]
        categories = np.unique(labels)

        n_explanations = len(explanations)

        print(probs.shape)
        print(labels)
        print(explanations)
        print(categories)

        ex_generator = ClassificationExampleGenerator(list(categories))
        for j, (nn_outcomes, label) in enumerate(zip(probs, labels)):
            ctxs_atoms = ['concept({}).' for i in range(len(nn_outcomes))]
            ex_generator.parse(nn_outcomes, ctxs_atoms, label)

        with open(out_file, 'w') as f:
            for example in ex_generator.get_examples(n_of_predicates=n_explanations):
                f.write(example)
                f.write('\n\n')
