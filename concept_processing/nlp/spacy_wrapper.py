"""
Adapter for the spacy library
"""

from typing import Iterator, Union

from spacy import tokens
import en_core_web_sm

from concept_processing.nlp.nlp_parser import NLPDocument, NLPToken, NLPTokenSpan


class SpacyToken:
    def __init__(self, token: tokens.Token):
        self._token = token

    def index(self) -> int:
        return self._token.i

    def lower(self) -> str:
        return self._token.lower_

    # More fine-grained set of tags then pos()
    def tag(self) -> str:
        return self._token.tag_

    def pos(self) -> str:
        return self._token.pos_

    def dep(self) -> str:
        return self._token.dep_

    def doc(self) -> 'NLPDocument':
        return SpacyDocument(self._token.doc)

    def children(self) -> Iterator['NLPToken']:
        for child in self._token.children:
            yield SpacyToken(child)

    def ancestors(self) -> Iterator['NLPToken']:
        for ancestor in self._token.ancestors:
            yield SpacyToken(ancestor)

    def __str__(self):
        return str(self._token)


class SpacyTokenSpan:
    def __init__(self, sent: tokens.Span):
        self._sent = sent
        self.iter = None

    def root(self) -> NLPToken:
        return SpacyToken(self._sent.root)

    def __iter__(self):
        self.iter = self._sent.__iter__()
        return self

    def __next__(self) -> NLPToken:
        return SpacyToken(self.iter.__next__())

    def __str__(self) -> str:
        return str(self._sent)

    def __getitem__(self, item) -> Union['NLPTokenSpan', NLPToken]:
        if isinstance(item, slice):
            return SpacyTokenSpan(self._sent[item])
        return SpacyToken(self._sent[item])


class SpacyDocument:
    def __init__(self, doc: tokens.Doc):
        self._doc = doc
        self.iter = None

    def sentences(self) -> Iterator[NLPTokenSpan]:
        for sent in self._doc.sents:
            yield SpacyTokenSpan(sent)

    def __len__(self) -> int:
        return len(self._doc)

    def __str__(self) -> str:
        return str(self._doc)

    def __iter__(self):
        self.iter = self._doc.__iter__()
        return self

    def __next__(self) -> NLPToken:
        return SpacyToken(self.iter.__next__())


class SpacyWrapper:
    def __init__(self):
        self._nlp = en_core_web_sm.load()

    def __call__(self, *args, **kwargs) -> NLPDocument:
        spacy_doc = self._nlp(*args, **kwargs)
        return SpacyDocument(spacy_doc)
