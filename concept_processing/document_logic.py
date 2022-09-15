from typing import Iterator, Tuple

import numpy as np

from concept_processing.nlp.nlp_parser import NLPToken


def iterate_dependencies(parent: NLPToken) -> Iterator[Tuple[NLPToken, NLPToken]]:
    for child in parent.children():
        yield (parent, child)
        for depend in iterate_dependencies(child):
            yield depend


def iterate_ancestors(curr: NLPToken) -> Iterator[NLPToken]:
    for pred in curr.ancestors():
        yield pred
        for depend in iterate_ancestors(pred):
            yield depend

def iterate_children(curr: NLPToken) -> Iterator[NLPToken]:
    for child in curr.children():
        yield child
        for depend in iterate_children(child):
            yield depend


# Match the best target token with the best token from the premise
def arg_best_match(anchor, toks, position_factor=1):
    """
    Compares anchor separately with each element of toks and determines which is closer.
    Returns index of closest token. If they match returns lowest such index
    Assumes that both have the same text. If tok[i].text != tok[j].text then all
    bets are off.
    """
    scores = [token_match_score(anchor, tok, position_factor) for tok in toks]
    besti = np.argmax(scores)
    if scores[besti] == 0:
        raise ValueError(f"Cannot find good match for {anchor.text}")
    return besti


def token_match_score(tok1: NLPToken, tok2: NLPToken, position_factor: int = 1) -> float:
    def eq(tok1: NLPToken, tok2: NLPToken) -> bool:
        t1 = tok1.lower()
        t2 = tok2.lower()
        if (t1 == 'a' and t2 == 'an') or \
                (t1 == 'an' and t2 == 'a'):
            return True

        return t1 == t2

    score = 0
    # checks if tokens are not equal taking into account edge cases such as a/an
    if not eq(tok1, tok2):
        return score
    score += 1
    if tok1.dep() != tok2.dep() and tok1.dep() != 'ROOT' and tok2.dep() != 'ROOT':
        return score
    score += 1
    score += token_ancestor_match(tok1, tok2)
    score += token_child_match(tok1, tok2)
    if tok1.index() == tok2.index():
        score += position_factor
    if (len(tok1.doc()) - tok1.index()) == (len(tok2.doc()) - tok2.index()):
        score += position_factor
    return score

def token_ancestor_match(tok1, tok2):
    i = -1
    for i, (a1, a2) in enumerate(zip(iterate_ancestors(tok1), iterate_ancestors(tok2))):
        if str(a1) != str(a2):
            return i
    return i+1

def token_child_match(tok1, tok2):
    i = -1
    for i, (a1, a2) in enumerate(zip(iterate_children(tok1), iterate_children(tok2))):
        if str(a1) != str(a2):
            return i
    return i+1

