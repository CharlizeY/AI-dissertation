from typing import List

import numpy as np

def create_labels_as_indices(label_strs: List[str]) -> (np.ndarray, np.ndarray):
    categories = np.unique(label_strs)
    K = categories.size
    category_dict = { cat: i for i, cat in enumerate(categories)}
    labels_as_indices = np.array([category_dict[label_str] for label_str in label_strs])
    return labels_as_indices, categories


def label_indices_to_one_hot(labels_as_indices: np.ndarray, K: int=None):
    if K is None:
        K = np.max(labels_as_indices) + 1
    labels = np.zeros((labels_as_indices.shape[0],K))
    labels[np.arange(labels_as_indices.shape[0]),labels_as_indices] = 1
    return labels


def calculate_concept_purity_measures(pam: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Concept purities is a simple mixture vector of the labels associated with that concept. That is to say, consider that your pam contains concept j in column j. Then every row i in which column j is a 1 is an example of that concept. And is associated with class label from labels[i]. This can be though of a sample from a categorical distribution over the class categories. If concept j appears Nj times in the pam then we have Nj samples from that categorical distribution. 

    parameters
    ----------
    pam - N x C presence/absence matrix
    labels - N x K - matrix of one-hot labels
    
    returns
    -------
    concept_label_counts - C x K matrix
        The output concept_label counts is a matrix where each row c is a vector of counts of labels corresponding ot concept c. And where element (c,k) is the number of times a datapoint with concept c occurs with label k.
    """
    N, C = pam.shape
    K = labels.shape[1]
#    concept_mean_labels = np.empty((C, K))
    # concept_label_counts counts the frequency of each label associated with said concept
    concept_label_counts = np.empty((C, K))
    for c in range(C):
        have_concept = pam[:,c] == 1
        #print(f"labels[have_concept,:] = {labels[have_concept,:]}")
        #print(f"np.mean(labels[have_concept,:], axis=0) = {np.mean(labels[have_concept,:], axis=0)}")
#        concept_mean_labels[c,:] = np.mean(labels[have_concept,:], axis=0)
        concept_label_counts[c, :] = np.sum(labels[have_concept,:], axis=0)
        #break
    concept_mean_labels = concept_label_counts/np.sum(concept_label_counts,axis=1).reshape(-1,1)
    return concept_label_counts, concept_mean_labels

def _calc_log_evidence(label_counts_i, alpha):
    """
    Using the beta-binomial model with symmetric prior, this calculates the
    log-evidence of the observed counts.
    
    label_counts_i - a K-vector of the counts of the labels 
    alpha - the symmetric prior (a scalar)
    """
    K = label_counts_i.size
    Ni = np.sum(label_counts_i)
    log_evidence = - Ni * np.log(Ni + K*alpha)
    log_evidence += np.sum([n_ik * np.log(n_ik + alpha) if n_ik > 0 else 0 for n_ik in label_counts_i])
    return log_evidence

def _calc_log_evidence_indiv_model(label_counts_i, label_counts_j, alpha):
    """
    Using the beta-binomial model with symmetric prior, this calculates the
    log-evidence of the observed counts for two different concepts i and j
    under the assumptionn that they come from separate distributions.
    
    label_counts_i - a K-vector of the counts of the labels for concept i 
    label_counts_i - a K-vector of the counts of the labels  for concept j
    alpha - the symmetric prior (a scalar)
    """
    log_evidence = _calc_log_evidence(label_counts_i, alpha)
    log_evidence += _calc_log_evidence(label_counts_j, alpha)
    return log_evidence

def _calc_log_evidence_combined_model(label_counts_i, label_counts_j, alpha):
    """
    Using the beta-binomial model with symmetric prior, this calculates the
    log-evidence of the observed counts for two different concepts i and j
    under the assumptionn that they come from the same distribution.
    
    label_counts_i - a K-vector of the counts of the labels for concept i 
    label_counts_i - a K-vector of the counts of the labels  for concept j
    alpha - the symmetric prior (a scalar)
    """
    log_evidence = _calc_log_evidence(label_counts_i + label_counts_j, alpha)
    return log_evidence

def _calc_log_evidence_ratio(label_counts_i, label_counts_j, alpha):
    """
    Using the beta-binomial model with symmetric prior, this calculates the
    log of the evidence ratio for the observed counts for two different concepts
    i and j given the two assumed models independent versus combined.
    
    label_counts_i - a K-vector of the counts of the labels for concept i 
    label_counts_i - a K-vector of the counts of the labels  for concept j
    alpha - the symmetric prior (a scalar)
    """
    log_ratio = _calc_log_evidence_indiv_model(label_counts_i, label_counts_j, alpha)
    log_ratio -= _calc_log_evidence_combined_model(label_counts_i, label_counts_j, alpha)
    return log_ratio
