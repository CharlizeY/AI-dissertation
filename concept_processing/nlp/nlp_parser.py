"""
Types for NLP library to allow easy change
"""

from typing import Protocol, Iterator, Union


class NLPToken(Protocol):
    def index(self) -> int:
        ...

    def lower(self) -> str:
        ...

    # More fine-grained set of tags then pos()
    def tag(self) -> str:
        ...

    def pos(self) -> str:
        ...

    def dep(self) -> str:
        ...

    def doc(self) -> 'NLPDocument':
        ...

    def children(self) -> Iterator['NLPToken']:
        ...

    def ancestors(self) -> Iterator['NLPToken']:
        ...

    def __str__(self):
        ...


class NLPTokenSpan(Protocol):
    def root(self) -> NLPToken:
        ...

    def __iter__(self):
        ...

    def __next__(self) -> NLPToken:
        ...

    def __str__(self) -> str:
        ...

    def __getitem__(self, item) -> Union['NLPTokenSpan', NLPToken]:
        ...


class NLPDocument(Protocol):
    def sentences(self) -> Iterator[NLPTokenSpan]:
        ...

    def __len__(self) -> int:
        ...

    def __str__(self) -> str:
        ...

    def __iter__(self):
        ...

    def __next__(self) -> NLPToken:
        ...


class NLPParser(Protocol):
    def __call__(self, *args, **kwargs) -> NLPDocument:
        ...
