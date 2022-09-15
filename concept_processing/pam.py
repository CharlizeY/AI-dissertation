from collections import Counter
from typing import List

import numpy as np
import sklearn


def count_datapoints_in_each_feature(pam):
    """
    Count the number of datapoints with each feature
    """
    return np.sum(pam, axis=0).astype(int)


def count_features_in_each_datapoint(pam):
    """
    Count the number of features in each datapoint
    """
    return np.sum(pam, axis=1).astype(int)


#
def prune_and_reindex_concepts(pam, concepts, threshold):
    """
    Removes columns from pam matrix below a given threshold frequency.
    We have K original concepts and K' with counts greater than or equal
    to `threshold`.
    
    parameter
    ---------
    pam - original presence/absence matrix with N rows (each a datapoint) and K columns (each a concepts)
    concepts - the list of K concepts
    threshold - an integer
     
    returns
    -------
    pruned_pam - a N x K' pam including all columns with frequency >=threshold
    pruned_concepts - a K' length sub-list of the original concepts corresponding 
        to the retained concepts/columns.
    inclusion_mapping - an inclusion function mapping from the new index of a 
        concept in the pruned list to the original index in the old list
    """
    concept_counts = np.sum(pam, axis=0).astype(int)
    filter_ = concept_counts >= threshold
    pruned_pam = pam[:, filter_]
    inclusion_mapping = []
    pruned_concepts = []
    for i, (concept, include) in enumerate(zip(concepts, filter_)):
        if include:
            pruned_concepts.append(concept)
            inclusion_mapping.append(i)
    return pruned_pam, pruned_concepts, inclusion_mapping


def prune_concepts_general(
        pam, method='by_threshold', label_ids=None, threshold=3, K=None,
        frac_threshold=None):
    """
    This prunes any pam and returns a pruned pam and a mapping from new indices
    to old indices.
    """
    if method == 'by_count_threshold':
        return prune_concepts_by_threshold(pam, threshold)
    elif method == 'by_independent_mi':
        return prune_concepts_by_independent_mi(pam, label_ids, K)
    elif method == 'by_cummulative_mi':
        return prune_concepts_by_cummulative_mi(
            pam, label_ids, threshold, frac_threshold=frac_threshold, maxK=K)
    else:
        raise ValueError(f'Unrecognised pruning method {method}')


def prune_concepts_by_threshold(pam, threshold):
    # N datapoints and K original concepts
    N, J = pam.shape
    concept_counts = count_datapoints_in_each_feature(pam)
    filter_ = (concept_counts >= threshold)
    # pruned matrix
    pruned_pam = pam[:, filter_]
    # new id mapping to old id
    pruned2orig = np.arange(J)[filter_]
    return pruned_pam, pruned2orig


def prune_concepts_by_independent_mi(pam, label_ids, K):
    """
    Keeps the first K concepts with the highest independent mutual information
    with the label. In other words, if L is the random variable of the label
    and C_j is the random variable of concept j, then mutual information
    I(L; C_j) is the independent mutual information calculated for concept j
    These are then sorted (largest first) and the first k concepts are kept.
    """
    if label_ids is None:
        raise ValueError('prune_concepts_by_independent_mi needs label_ids')
    N, J = pam.shape
    mi_scores = calc_independent_mis(pam, label_ids)
    reorder = np.argsort(mi_scores)[::-1][:K]
    pruned_pam = pam[:, reorder]
    pruned2orig = reorder
    return pruned_pam, pruned2orig


def calc_independent_mis(pam, label_ids):
    """
    Best done with merged pam and merged label ids
    """
    N, J = pam.shape
    mi_scores = np.array([
        sklearn.metrics.mutual_info_score(pam[:, j], label_ids) for j in range(J)])
    return mi_scores


def prune_concepts_by_cummulative_mi(pam, label_ids, threshold, frac_threshold=None, maxK=None):
    """
    Greedily seek the first K concepts which explain frac of the total mutual
    information with the label ids.
    In other words, if L is the random variable of the label
    and C_j is the random variable of concept j then C_{j_1} is the concept that
    explains the most MI with the label alone, i.e.
      j_1 = argmax_j I(L;C_j)
    And subsequent concepts are chosen one at a time so that they explain as
    much additional MI given that the previous concepts are already known
    So C_{j_2} is the concept that conditioned on C_{j_1} explains the most
    additional mutual information, i.e.
    j_2 = argmax_{j} I(L;C_j | C_{j_1})
    And 
    j_{k+1} = argmax_{j} I(L; C_j | C_{j_1}, ..., C_{j_k})
    
    
    We use:
    I(L;C_j | C_{j_1}, ..., C_{j_k}) 
      = I(L; C_j, C_{j_1}, ..., C_{j_k}) - I(L; C_{j_1}, ..., C_{j_k})
    And for joint RVs we simply reduce to an integer Z
    
    So for 
      C_{j_1}, ..., C_{j_k} we use integer
     Z = 2^(k-1)*C_{j_1} + ... + 2^0 * C_{j_k}
     
    And for some arbitrary j
    for C_j, C_{j_1}, ..., C_{j_k} we use
    Z_j = 2^k*C_{j_1} + ... + 2 * C_{j_k} + 2^0 * C_j
    
    So our equation becomes    
    I(L;C_j | Z ) 
      = I(L; Z_j ) - I(L; Z)

    """
    if label_ids is None:
        raise ValueError('prune_concepts_by_cummulative_mi needs label_ids')
    cum_mis, reorder = calc_cummulative_mi(pam, label_ids, threshold, maxK=maxK)
    # normalise by the maximum
    cum_mis /= cum_mis[-1]
    # so now the conditioning concepts are those columns kept
    # threshold by the desired fraction of the overall MI
    if not maxK is None:
        pruned2orig = reorder[:maxK]
    else:
        raise NotImplemented("Need to introduce early stopping")
        pruned2orig = reorder[cum_mis < frac_threshold]
    pruned_pam = pam[:, pruned2orig]
    return pruned_pam, pruned2orig


def convert_rows_to_unique_integers(pam, included_columns):
    """
    For a subset of columns convert the corresponding rows to identifying integers
    """
    rows_as_strs = np.array(list(map(str, pam[:, included_columns]))).reshape((-1, 1))
    unique_strs = np.unique(rows_as_strs).reshape((1, -1))
    rows_as_ints = np.where((rows_as_strs == unique_strs))[1]
    return rows_as_ints


def calc_cummulative_mi(pam, label_ids, threshold, maxK=None, verbose=True):
    """
    threshold specifies frequency below which to ignore concepts for simpler
    calculation
    maxK specifies the stopping criterion if requested in terms of the maximum number
    of concepts to include. If this is higher than the number of concepts after
    thresholding then a lower number of concepts is returned.
    """
    N, J = pam.shape
    indep_mi_scores = calc_independent_mis(pam, label_ids)
    j_1 = np.argmax(indep_mi_scores)
    condition_Js = [j_1]
    Z = convert_rows_to_unique_integers(pam, [j_1])
    cum_mis = [indep_mi_scores[j_1]]
    # by setting a threshold we exclude some very rare concepts from calculation
    included_Js = [j for j in range(J) if np.sum(pam[:, j]) >= threshold]
    if maxK is None:
        K = len(included_Js)
    else:
        K = min(maxK, len(included_Js))
    for k in range(1, K):
        if verbose:
            print(f".", end='')
        post_mi_scores = calc_post_mis(pam, label_ids, condition_Js, Z, included_Js)
        j_k = np.argmax(post_mi_scores)
        condition_Js.append(j_k)
        # next cummulative mutual information is last plus the conditional of this
        cum_mis.append(post_mi_scores[j_k])
        Z = convert_rows_to_unique_integers(pam, condition_Js + [j_k])
    cum_mis = np.array(cum_mis)
    reorder = np.array(condition_Js)
    return cum_mis, reorder


def calc_post_mis(pam, label_ids, condition_Js, preZ, included_Js):
    N, J = pam.shape
    # pre_mi = sklearn.metrics.mutual_info_score(Z, label_ids)
    post_mi_scores = np.zeros(J)
    noncondition_Js = [j for j in included_Js if not j in condition_Js]
    for j in noncondition_Js:
        # Z_j = convert_rows_to_unique_integers(pam, condition_Js + [j])
        # for efficiency we use simplified recipe to give an integer representation
        # of the concept RV of all conditions and new column j
        Z_j = 2 * preZ + pam[:, j]
        post_mi = sklearn.metrics.mutual_info_score(Z_j, label_ids)
        post_mi_scores[j] = post_mi
    return post_mi_scores


def filter_pruned_groups(pruned2orig, listoflists):
    newlistoflists = [listoflists[oldid] for oldid in pruned2orig]
    return newlistoflists


def group_concepts(pam: np.ndarray, concepts: List[str], concept2group: np.ndarray):
    """
    parameters
    ----------
    concepts (list<str>[K]) - a list of K concepts as strings 
      (treated as a mapping from id to string)
    concept2group (array[K]) - a mapping from concept ids to group id this 
      assumes that the group ids are contiguous and have 0 minimum

    Returns
    -------
    grouped_pam - a pam where each column is a group of concepts
    grouped_concept_ids - List[List[int]], for each group a list of concept ids
    grouped_concepts - List[List[str]] for each group concept a list of the concepts in that group.
    grouped_concept_counts - List[int] a total count of the number of covered instances from a group concept.
    max_child_counts - List[int] of the contained concepts in a given group what is the max number of instances covered by a child concept.
    dominant_concepts - List[str] the string of the most frequent child concept in each group.
    """
    n_groups = np.max(concept2group) + 1
    groups = np.arange(n_groups)
    concept_counts = count_datapoints_in_each_feature(pam)
    if np.min(concept2group) != 0:
        raise ValueError(f"Minimum id of group is {np.min(concept2group)}. Should be 0.")
    # concept groups maps from original concept id to group
    # concepts as original ids grouped by group id
    grouped_concept_ids = [list() for _ in groups]
    # concepts as strings grouped by group id
    grouped_concepts = [list() for _ in groups]
    grouped_concept_counts = np.zeros(n_groups, dtype=int)
    max_child_counts = np.zeros(n_groups, dtype=int)
    # each group id has a dominant label (the one with the largest frequency count) 
    dominant_concepts = [''] * n_groups
    # grouped pam
    grouped_pam = np.zeros((pam.shape[0], n_groups), dtype=int)
    for concept_id, group_id in enumerate(concept2group):
        # keep track of dominant labels and child counts
        child_count = concept_counts[concept_id]
        child_label = concepts[concept_id]
        if max_child_counts[group_id] < child_count:
            max_child_counts[group_id] = child_count
            dominant_concepts[group_id] = child_label
        grouped_concept_counts[group_id] += child_count
        grouped_concept_ids[group_id].append(concept_id)
        grouped_concepts[group_id].append(child_label)
        grouped_pam[:, group_id] |= pam[:, concept_id]
    #
    return dict(grouped_pam=grouped_pam,
                grouped_concept_ids=grouped_concept_ids,
                grouped_concepts=grouped_concepts,
                grouped_concept_counts=grouped_concept_counts,
                dominant_concepts=dominant_concepts,
                max_child_counts=max_child_counts)


def merge_datapoints_by_id(ids, pam):
    unique_ids = np.unique(ids)
    merged_pam = np.zeros((unique_ids.size, pam.shape[1]), dtype=int)
    for i, uid in enumerate(unique_ids):
        for j in np.where(ids == uid)[0]:
            merged_pam[i, :] |= pam[j, :]

    return unique_ids, merged_pam

def merge_datapoints_by_id_w_labels(ids, pam, labels_as_indices):
    unique_ids = np.unique(ids)
    merged_pam = np.zeros((unique_ids.size, pam.shape[1]), dtype=int)
    merged_labels = np.zeros(unique_ids.size, dtype=int)
    conversion_dict = {}
    for i, uid in enumerate(unique_ids):
        original_indices = np.where(ids == uid)[0]
        for j in original_indices:
            conversion_dict[j] = i
            merged_pam[i, :] |= pam[j, :]
        merged_labels[i] = Counter(labels_as_indices[original_indices]).most_common()[0][0]
    return unique_ids, merged_pam, merged_labels

def get_merged_label_ids(labels, ids):
    # we want the labels to be merged by labels too
    merged_label_ids = merge_datapoints_by_id(ids, labels.reshape((-1, 1)))[1].flatten()
    return merged_label_ids
