import itertools
import random
from typing import List

# Parses all sentences immediately instead of 1 by 1.
from concept_processing.asp.asp_generator import ASPGenerator
from concept_processing.enums import ProblemType
from concept_processing.nlp.nlp_parser import NLPParser


# A wrapper around the ASPGenerator which allows parsing of lists immediately instead of
# individual elements
class BatchASPGenerator:
    def __init__(self, spacy_lang: NLPParser, problem_type: ProblemType):
        self.asp_generator = ASPGenerator(spacy_lang, problem_type)
        self.programs = []
        self.examples = []

        # For reproducibility
        random.seed(0)

    def parse(self, premise_texts: List[str], concepts_texts: List[str] = None):
        self.programs = []
        self.examples = []

        if concepts_texts is None:
            concepts_texts = itertools.repeat(None)

        for i, (premise_text, concepts_text) in enumerate(zip(premise_texts, concepts_texts)):
            self.asp_generator.parse(premise_text, concepts_text)
            self.programs.append(self.asp_generator.get_programs())
            self.examples.append(self.asp_generator.get_examples(i))

    def get_programs(self) -> List[List[str]]:
        return [item for sublist in self.programs for item in sublist]

    # Length of the list is the number of folds.
    def get_examples(self, n_folds=1) -> List[str]:
        random.shuffle(self.examples)

        return ['\n'.join(self.examples[i::n_folds]) for i in range(n_folds)]
