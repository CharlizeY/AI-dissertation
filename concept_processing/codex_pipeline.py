"""
Builds the pipeline for extracting concepts from text.
The stages are described in the report
"""

from typing import Dict, List, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from concept_processing import io
from concept_processing.concepts import build_embedding_matrix
from concept_processing.extraction import ConceptsState
from concept_processing.grouping import calc_prox_mtx_labels, calc_prox_mtx_embedding
from concept_processing.labels import create_labels_as_indices, label_indices_to_one_hot, \
    calculate_concept_purity_measures
from concept_processing.nlp.spacy_wrapper import SpacyWrapper
from concept_processing.pam import prune_concepts_general, group_concepts, merge_datapoints_by_id, get_merged_label_ids

hyperparameters = {
    "new": {
        "best_metric": 'manhattan',
        "best_model": 'stsb-roberta-base',
        "best_alpha": 0.5,  # 0.5
        "best_threshold": 145,  # 50 # 150 # 145
        "best_linkage": 'single',
        "best_lambda": 0.4,  # 1 # 0.4 # 0.1
        "best_labelmetric": 'beta_ratio',  # evidence_ratio

    },
    "old": {
        "best_metric": 'manhattan',
        "best_model": 'stsb-roberta-base',
        "best_alpha": 0.5,  # 0.5
        "best_threshold": 210,  # 210 # 220 best from Luke's code
        "best_linkage": 'single',
        "best_lambda": 0.4,  # 0.2,
        "best_labelmetric": 'evidence_ratio',

    }
}


def _from_new_2_old_index(curr: ConceptsState, pam: np.ndarray, new_2_old_id: np.ndarray) -> Tuple[
        ConceptsState, Dict[int, int]]:
    new_concept_strings = [curr.concept_strings[old_id] for old_id in new_2_old_id]

    old2new = {old_i: new_i for new_i, old_i in enumerate(new_2_old_id)}
    return ConceptsState(curr.ids, curr.label_indices, curr.label_categories, pam, new_concept_strings), old2new


# Extracts concept sentences using atomisation followed by generalisation
def extract_concepts(examples_dir: str) -> ConceptsState:
    nlp = SpacyWrapper()
    concept_bag, concept_list = io.capture_all_concepts(nlp, examples_dir)
    label_indices, categories = create_labels_as_indices(concept_bag.labels)
    return ConceptsState(concept_bag.ids, label_indices, categories, concept_bag.to_pam(), concept_list)


def simple_pruning(concepts_state: ConceptsState, hyperparameters: Dict[str, Any]) -> Tuple[ConceptsState, Dict[int, int]]:
    simple_prune_pam, simple_prune_2_raw_id = prune_concepts_general(concepts_state.concept_pam,
                                                                     method='by_count_threshold', threshold=2)

    return _from_new_2_old_index(concepts_state, simple_prune_pam, simple_prune_2_raw_id)


# def separate(concepts_state: ConceptsState, hyperparameters: Dict[str, Any]) -> Tuple[ConceptsState, Dict[int, int]]:
#     return None


# def labelling(concepts_state: ConceptsState, hyperparameters: Dict[str, Any]) -> Tuple[ConceptsState, Dict[int, int]]:
#     return None


def grouping(concepts_state: ConceptsState, hyperparameters: Dict[str, Any]) -> Tuple[ConceptsState, Dict[int, int]]:
    label_one_hot = label_indices_to_one_hot(concepts_state.label_indices)

    sentence_transfomer = SentenceTransformer(hyperparameters["best_model"])
    # matrix of embedding vectors over the concepts
    emb_matrix = build_embedding_matrix(concepts_state.concept_strings, sentence_transfomer)
    embedding = calc_prox_mtx_embedding(emb_matrix, metric=hyperparameters["best_metric"])
    prox_mtx_embedding = embedding

    # TODO: calculate_concept_purity_measures looks like it does not need label_one_hot,
    #  i.e. can work with label_indices with no issues
    label_counts_mtx, _ = calculate_concept_purity_measures(concepts_state.concept_pam, label_one_hot)
    prox_mtx_labels = calc_prox_mtx_labels(label_counts_mtx, alpha=hyperparameters["best_alpha"],
                                           labelmetric=hyperparameters["best_labelmetric"])
    prox_mtx = prox_mtx_embedding + hyperparameters["best_lambda"] * prox_mtx_labels

    model = AgglomerativeClustering(distance_threshold=hyperparameters["best_threshold"], n_clusters=None,
                                    linkage=hyperparameters["best_linkage"],
                                    affinity='precomputed').fit(prox_mtx)

    group_concepts_results = group_concepts(concepts_state.concept_pam.astype(int), concepts_state.concept_strings,
                                            model.labels_)
    grouped_pam = group_concepts_results['grouped_pam']
    grouped_concept_ids = group_concepts_results['grouped_concept_ids']
    dominant_concepts = group_concepts_results['dominant_concepts']

    old2newid = {old_id: new_id for new_id, group_ids in enumerate(grouped_concept_ids) for old_id in group_ids}
    return ConceptsState(concepts_state.ids, concepts_state.label_indices, concepts_state.label_categories, grouped_pam,
                         dominant_concepts), old2newid


def pruning(concepts_state: ConceptsState, hyperparameters: Dict[str, Any]) -> Tuple[ConceptsState, Dict[int, int]]:
    pruned_grouped_pam, pruned2groupid = prune_concepts_general(concepts_state.concept_pam, method='by_cummulative_mi',
                                                                label_ids=concepts_state.label_indices, K=500, #K=78 # 300
                                                                threshold=2, #threshold=3
                                                                frac_threshold=0.9) #frac_threshold=0.9 # 0.5

    return _from_new_2_old_index(concepts_state, pruned_grouped_pam, pruned2groupid)


class CodexPipeline:
    def __init__(self, methods: List[str] = None, use_old_pipeline=False):
        if methods is None and not use_old_pipeline:
            methods = ['simple_pruning', 'grouping', 'pruning']
        if methods is None and use_old_pipeline:
            methods = ['grouping', 'pruning', 'id_merging']
        self.pipeline = []
        for method in methods:
            if method == 'simple_pruning':
                self.pipeline.append(simple_pruning)
            elif method == 'grouping':
                self.pipeline.append(grouping)
            elif method == 'pruning':
                self.pipeline.append(pruning)
            # elif method == 'id_merging':
            #     self.pipeline.append(merge_by_id)

        self.hyperparameters = hyperparameters["old" if use_old_pipeline else "new"]
        self.use_old_pipeline = True

    def __call__(self, concepts_state: ConceptsState) -> Tuple[ConceptsState, List[Dict[int, int]]]:
        conversion_dict_list = []
        for method in self.pipeline:
            concepts_state, conversion_dict = method(concepts_state, self.hyperparameters)
            conversion_dict_list.append(conversion_dict)
        return concepts_state, conversion_dict_list
