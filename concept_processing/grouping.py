from typing import Any

import numpy as np
from scipy.special import loggamma
from sklearn.metrics import pairwise_distances

from concept_processing.labels import _calc_log_evidence_ratio


# Calcluate the embedding distances
def calc_prox_mtx_embedding(embeds: np.ndarray, metric: str = 'manhattan', p: int = 768):
    # embedding distances    
    if metric == 'manhattan':
        prox_mtx_embedding = pairwise_distances(embeds, metric=metric)
    elif metric == 'cosine':
        prox_mtx_embedding = pairwise_distances(embeds, metric=metric)
    elif metric == 'minkowski':
        ## for minkowski we have to clean up the non-finite values
        prox_mtx_embedding = pairwise_distances(embeds, metric=metric, p=768)
        dummy_dist = np.max(prox_mtx_embedding[np.isfinite(prox_mtx_embedding)]) * 1.1
        prox_mtx_embedding[~np.isfinite(prox_mtx_embedding)] = dummy_dist
    elif metric == 'chebyshev':
        prox_mtx_embedding = pairwise_distances(embeds, metric=metric)
    else:
        raise ValueError(f"Unrecognised metric: {metric}")
    return prox_mtx_embedding


# Calculate the label distances
def calc_prox_mtx_labels(label_counts_mtx: np.ndarray, alpha: float = 0.5, labelmetric: str = 'evidence_ratio'):
    if labelmetric == 'evidence_ratio':
        return _calc_prox_mtx_labels_evidence_ratio(label_counts_mtx, alpha)
    elif labelmetric == 'beta_ratio':
        return _calc_prox_mtx_labels_beta_ratio(label_counts_mtx, alpha)
    else:
        raise ValueError(f"Unrecognised label metric {labelmetric}")


def _calc_prox_mtx_labels_evidence_ratio(label_counts_mtx, alpha: float = 0.5) -> np.ndarray:
    prox_mtx_labels = np.array([[
        np.exp(_calc_log_evidence_ratio(rowi, rowj, alpha)) for rowj in label_counts_mtx]
        for rowi in label_counts_mtx])
    return prox_mtx_labels


def _calc_prox_mtx_labels_beta_ratio(label_counts_mtx, alpha=0.5):
    prox_mtx_labels = np.array([[
        np.exp(_calc_log_beta_ratio(rowi, rowj, alpha)) for rowj in label_counts_mtx]
        for rowi in label_counts_mtx])
    return prox_mtx_labels


def _calc_log_beta_ratio(n, m, alpha):
    log_ratio = _calc_log_multidim_beta(n, alpha)
    log_ratio += _calc_log_multidim_beta(m, alpha)
    log_ratio -= _calc_log_multidim_beta(n + m, alpha)
    return log_ratio


def _calc_log_multidim_beta(n, alpha):
    return np.sum(loggamma(n + alpha)) - loggamma(np.sum(n + alpha))
