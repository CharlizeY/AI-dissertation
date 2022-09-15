import re
import string
from typing import List, Dict

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

from concept_processing.io import get_datapoint_iterator
from concept_processing.pam import count_datapoints_in_each_feature



# Calculates the distances between concepts using a transformer
def build_embedding_matrix(concepts: List[str], transformer_model: SentenceTransformer,
                           include_articles: bool = False) -> np.ndarray:
    # build embedding matrix
    C = len(concepts)
    embeds = np.empty((C, 768))
    for i, concept in enumerate(concepts):
        if not include_articles:
            concept = _remove_articles(concept)
        embeds[i, :] = transformer_model.encode(concept)
    return embeds


def _remove_articles(text: str):
    """
    We remove articles (and other troublesome words) that interfere with the
    clustering.
    """
    articles = {'a', 'an', 'the'}
    rest = [word for word in text.split() if not word.lower() in articles]
    if rest[0].lower() == 'then':
        rest.pop(0)
    return ' '.join(rest)


def _get_concept_indices_by_substring(sentstr, grouped_concept_ids, grouped_concepts):
    """
    Helper method for complete_pam_with_substrings
    """
    thesegroupids = []
    for groupid, (rawids, rawstrings) in enumerate(zip(grouped_concept_ids, grouped_concepts)):
        for rawid, rawstring in zip(rawids, rawstrings):
            if rawstring in sentstr:
                thesegroupids.append(groupid)
    return thesegroupids
