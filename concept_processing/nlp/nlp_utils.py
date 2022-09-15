import re

# Reverts the lower case sentence to its true case.
from concept_processing.nlp.nlp_parser import NLPParser


# Truecase words which are proper nouns and the starts of the sentences
def truecase(inp: str, spacy_lang: NLPParser) -> str:
    text_graph = spacy_lang(inp)

    sents = []
    for sent in text_graph.sentences():
        # Capitalise proper nouns
        normalized_sent = [str(tok).capitalize() if tok.tag() in ["NNP", "NNPS"] else str(tok) for tok in sent]
        normalized_sent[0] = normalized_sent[0].capitalize()
        reconstructed = re.sub(" (?=[.,'!?:;])", "", ' '.join(normalized_sent))
        sents.append(reconstructed)

    return " ".join(sents)


# Just capitalise the start of the sentence
def simple_truecase(inp: str) -> str:
    inp = inp.capitalize()
    return inp



def remove_punctuation(inp: str) -> str:
    return inp.strip('.!,?')


def add_punctuation(inp: str) -> str:
    if inp[-1] == '.':
        return inp
    return inp + '.'


# Merge n't with the word before: The hitter did n't swing -> The hitter didn't swing.
def merge_not(inp: str) -> str:
    tokens = inp.split()

    not_indices = [i for i, tok in enumerate(tokens) if tok == "n't"]
    for not_ind in not_indices:
        tokens[not_ind - 1:not_ind + 1] = [''.join(tokens[not_ind - 1:not_ind + 1])]

    return ' '.join(tokens)
