"""
File containing all deleted methods which are only used to evaluate w.r.t. previous code.
Only added capture_all_concepts_full_old to allow interfacing with new code
Not edited at all.
"""
import re
import benepar
import spacy
import nltk
# benepar.download('benepar_en3')

import numpy as np
from spacy.matcher import Matcher

from concept_processing.extraction import concept_dict_to_list, ConceptsState
from concept_processing.io import get_datapoint_iterator
from concept_processing.labels import create_labels_as_indices
from concept_processing.nlp.spacy_wrapper import SpacyWrapper
from concept_processing.pam import count_datapoints_in_each_feature, merge_datapoints_by_id, get_merged_label_ids, \
    merge_datapoints_by_id_w_labels


def capture_all_concepts_full_old(datapath: str) -> ConceptsState:
    nlp = SpacyWrapper()

    incmatcher, excmatcher = add_inc_and_exc_matchers(nlp)

    concept_dict, ids, label_strs, raw_bofs = capture_all_concepts_old(nlp, datapath, incmatcher, excmatcher)

    # now we convert the raw bag of concepts list-of-lists
    # to a presence absence matrix (old_pam)
    old_pam = convert_raw_bof_to_pam(raw_bofs, C=len(concept_dict))
    print(f"data.shape = {old_pam.shape}")
    print(f"We have {old_pam.shape[0]} datapoints with {old_pam.shape[1]} independent concepts.")

    old_concept_list = concept_dict_to_list(concept_dict)

    ## convert the labels to one hot vectors
    # there is one odd label that shouldn't be there. We change this to 'strike'.
    label_strs[
        label_strs == ' it could be called a strike because the pitch landed in the strike zone before being hit'] = 'strike'
    old_labels_as_indices, categories = create_labels_as_indices(label_strs)

    rows_to_remove = np.where(label_strs == 'none')[0]
    _, cols_to_remove = np.where(old_pam[
                                 label_strs == 'none', :])
    concepts_to_remove = [old_concept_list[col] for col in cols_to_remove]

    print(f"rows_to_remove = {rows_to_remove}")
    print(f"concepts_to_remove = {concepts_to_remove}")

    remove_none_labels = True
    if remove_none_labels:
        print("Before removing none label")
        print(f"pam.shape = {old_pam.shape}")
        print(f"len(concepts) = {len(old_concept_list)}")
        print(f"ids.shape = {ids.shape}")
        print(f"labels_as_indices.shape = {old_labels_as_indices.shape}")
        print(f"categories = {categories}")
        rows_to_remove = (label_strs == 'none')
        old_pam = old_pam[~rows_to_remove, :]
        cols_to_remove = (count_datapoints_in_each_feature(old_pam) == 0)
        old_pam = old_pam[:, ~cols_to_remove]
        old_concept_list = [concept for concept, to_remove in zip(old_concept_list, cols_to_remove) if not to_remove]
        label_strs = label_strs[~rows_to_remove]
        old_labels_as_indices, categories = create_labels_as_indices(label_strs)
        ids = ids[~rows_to_remove]
        resultsdict = dict(concepts=old_concept_list, ids=ids, categories=categories,
                           labels_as_indices=old_labels_as_indices, pam=old_pam)
        resultsdict['rows_to_remove'] = rows_to_remove
        # labels_as_indices = labels_as_indices[~rows_to_remove]
        # categories = categories[categories != 'none']
        print("After removing none label")
        print(f"pam.shape = {old_pam.shape}")
        print(f"len(concepts) = {len(old_concept_list)}")
        print(f"ids.shape = {ids.shape}")
        print(f"labels_as_indices.shape = {old_labels_as_indices.shape}")
        print(f"categories = {categories}")

    old_pam = old_pam.astype(int)
    unique_ids, merged_pam, labels_as_indices = merge_datapoints_by_id_w_labels(ids, old_pam, old_labels_as_indices)

    return ConceptsState(unique_ids, labels_as_indices, categories, merged_pam, old_concept_list)


def capture_all_concepts_old(nlp, path, incmatcher, excmatcher):
    """
    Iterate over all files over each contribution line in files.
    Convert to concepts and index these in concept dictionary.
    Store text as collection of concept ids.
    Returns concept_dict (text to id mapping) and rawbagofconcepts (list of lists of concept ids).
    """
    counter = 0
    concept_dict = {}
    rawbagofconcepts = []
    labels = []
    ids = []
    for id_, label, text in get_datapoint_iterator(path):
        try:
            # we remove double whitespace as it breaks the benepar plugin
            text = text.replace('�', '')
            text = re.sub(' +', ' ', text)
            text = ' '.join(text.split('\n'))
            if text[-1] not in ['!', ',', '.', '\n']:
                text += '.'
            doc = nlp(text)
        except:
            print(f"text = {text}")
            raise

        docconcepts, counter = extract_concepts(doc._doc, concept_dict, counter, incmatcher, excmatcher)
        rawbagofconcepts.append([id_, docconcepts])
        labels.append(label)
        ids.append(id_)
    return concept_dict, np.array(ids), np.array(labels), rawbagofconcepts


def extract_concepts(doc, concept_dict, counter, incmatcher, excmatcher):
    """
    Helper method for capturing concepts. Takes a single text and an existing dictionary
    and identifies existing concepts as well as new concepts which it adds to the dictionary
    """
    docconcepts = []
    for concept in iterate_concepts(doc, incmatcher, excmatcher):
        concept_str = re.sub(r"[^\w'\s]", '', concept.text.lower())
        if concept_str in concept_dict:
            cid = concept_dict[concept_str]
        else:
            cid = counter
            concept_dict[concept_str] = cid
            counter += 1
        docconcepts.append(cid)
    return docconcepts, counter


def iterate_concepts(doc, incmatcher, exclmatcher, maxlen=np.inf):
    """
    iterate over concepts in multi sentence document
    """
    for sent in doc.sents:
        # print(f"sent.text = {sent.text}")
        # print(f"sent = {sent}")
        count = 0
        for concept in iterate_concepts_in_span(sent, incmatcher, exclmatcher, maxlen):
            yield concept
            count += 1
        if count == 0 and len(sent) < maxlen and is_concept(sent, incmatcher, exclmatcher):
            yield sent[:-1]


def iterate_concepts_in_span(span, incmatcher, exclmatcher, maxlen=np.inf):
    """
    iterate over concepts in span, and recursively call on sub-spans
    """
    for child in span._.children:
        # print(f"child.text = {child.text}")
        count = 0
        for concept in iterate_concepts_in_span(child, incmatcher, exclmatcher, maxlen):
            yield concept
            count += 1
        # only yield child if sub-constituents do not match as concepts
        if count == 0 and len(span) < maxlen and is_concept(child, incmatcher, exclmatcher):
            yield child


def is_concept(span, incmatcher, exclmatcher):
    incmatches = len(list(incmatcher.__call__(span)))
    exclmatches = len(list(exclmatcher(span)))
    # print(f"span, matches = {(span.text, matches)}")
    return (incmatches > 0) and (exclmatches == 0)


def add_inc_and_exc_matchers(nlp):
    # create inclusion and exclusion matcher for our concept of concept
    incpattern1 = [{"POS": {"IN": ["NOUN", "PRON"]}}, {"POS": "AUX", "OP": "?"}, {"POS": "PART", "OP": "?"},
                   {"POS": "VERB"}]
    incpattern2 = [{"POS": {"IN": ["NOUN", "PRON"]}}, {"TEXT": {"IN": ["'s", "was", "were", "is", "are"]}}, {"OP": "+"}]
    excpattern = [{"POS": "SCONJ"}]
    
    # Dirty way of using SpacyWrapper, kept because this code is not maintained
    incmatcher = Matcher(nlp._nlp.vocab)
    excmatcher = Matcher(nlp._nlp.vocab)
    #
    if spacy.__version__.startswith('2'):
        incmatcher.add("action/event concept 1", None, incpattern1)
        # pattern = [{"POS":"VERB"},{"OP":"*"},{"POS": "NOUN"}]
        # matcher.add("verb concept 3", None, pattern)
        # pattern = [{"POS":"VERB"},{"OP":"*"},{"POS": "PRON"}]
        # matcher.add("verb concept 4", None, pattern)
        incmatcher.add("state concept 1", None, incpattern2)
        excmatcher.add("compound concept", None, excpattern)
        nlp._nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        incmatcher.add("action/event concept 1", [incpattern1])
        incmatcher.add("state concept 1", [incpattern2])
        excmatcher.add("compound concept", [excpattern])
        nlp._nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    return incmatcher, excmatcher


def convert_raw_bof_to_pam(raw_bags_of_features, C=None):
    """
    Takes a list of lists of ids and converts to binary presence-absence matrix (PAM).

    parameters
    ----------
    raw_bags_of_features
    C - maximum number of ids
    """
    N = len(raw_bags_of_features)
    if C is None:
        C = 0
        for bof in raw_bags_of_features:
            C = np.max(bof)
            C = max(C, thismax)
        C += 1
    data = np.zeros((N, C))
    for i, (id_, bof) in enumerate(raw_bags_of_features):
        data[i, bof] = 1
    return data
