from typing import List, Callable

from concept_processing.document_logic import arg_best_match, iterate_dependencies
from concept_processing.enums import ProblemType
from concept_processing.nlp.nlp_parser import NLPParser, NLPDocument, NLPToken


# Generates ASP programs as well as the examples for the atomisation/generalisation tasks
class ASPGenerator:
    def __init__(self, spacy_lang: NLPParser, problem_type: ProblemType):
        self._nlp = spacy_lang
        self.terms = {}
        self.inclusions = []
        self.max_concept_id = -1
        self._premise_id = 0

        self.predicate_fun = get_predicate_function(problem_type)
        self.merge_phrases = problem_type == ProblemType.ATOMISATION

        # Terms included in the context of an example and in all of the solutions.
        self.terms_used = ["tokens", "deps"]

    @staticmethod
    # . can be difficult to handle during learning. It's easier not to include it.
    def _skip_token(tok: NLPToken) -> bool:
        return str(tok) in ['.']

    def parse(self, premise_text: str, concepts_text: str = None):
        premise = self._nlp(premise_text)
        self._parse_premise(premise)
        if concepts_text is not None:
            concepts = self._nlp(concepts_text)
            premise_toks = list(filter(lambda t: not self._skip_token(t), premise))
            self._parse_concepts(concepts, premise_toks)

    def _parse_premise(self, premise: NLPDocument):
        self.terms = {}
        if len(premise) == 0:
            raise ValueError("Empty premise.")
        premise_id = self._premise_id
        for sent in premise.sentences():
            # print(premise_id)
            # print(self._premise_id)
            if premise_id > self._premise_id:
                raise ValueError("Only works with single sentence premises")

            self.terms["tokens"] = list()
            self.terms["sequence"] = list()
            self.terms["deps"] = list()
            self.terms["pos"] = list()
            for tok in sent:
                if not self._skip_token(tok):
                    self.terms["tokens"].append(predicate_token(tok.index(), f'"{tok.lower()}"'))
                    self.terms["pos"].append(predicate_pos_token(tok.index(), tok.pos().lower()))
            self.terms["sequence"].append(predicate_sentence_start(sent[0].index()))
            for a, b in zip(sent[:-1], sent[1:]):
                self.terms["sequence"].append(predicate_sentence_successor(a.index(), b.index()))
            root = sent.root()
            self.terms["deps"].append(predicate_doc_root(root.index()))
            for parent, child in iterate_dependencies(root):
                if not self._skip_token(parent) and not self._skip_token(child):
                    self.terms["deps"].append(predicate_doc_dependency(parent.index(), child.index(), child.dep()))
            # premise_id += 1

    def _parse_concepts(self, concepts: NLPDocument, premise_toks: List[NLPToken]):
        self.inclusions = []
        self.max_concept_id = len(premise_toks)

        for i, sent in enumerate(concepts.sentences()):
            try:
                concept_ref_ids = {tok.index(): arg_best_match(tok, premise_toks) for tok in sent if
                                   not self._skip_token(tok)}
            except ValueError:
                premise_sent = ' '.join([str(w) for w in premise_toks])
                print(f"Error encountered with premise:\n\t{premise_sent}\nand concept sentence\n\t{sent}.")
                raise

            self.inclusions.append(list(concept_ref_ids.values()))

    # generates multiple programs in case multiple concept sentences exist
    def get_programs(self) -> List[List[str]]:
        always_included = []
        for key, val in self.terms.items():
            if key in self.terms_used:
                always_included += val

        if "concept_tokens" not in self.terms:
            return [always_included]

        concept_inclusions = [[self.predicate_fun(id) for id in inc] for inc in self.inclusions]
        return [always_included + concepts for concepts in concept_inclusions]

    def get_examples(self, ex_num: int) -> str:
        context_terms = {included_term: self.terms[included_term] for included_term in self.terms_used}
        context = [item for sublist in context_terms.values() for item in sublist]
        context_str = '\n'.join(context)

        examples = []
        # for i, curr_inc_ids in enumerate(powerset(range(0, self.max_concept_id))):
        for i, curr_inc_ids in enumerate(self.inclusions):
            curr_inc_ids = set(curr_inc_ids)
            curr_exc_ids = set(range(0, self.max_concept_id)) - curr_inc_ids

            inclusion_str = ','.join([self.predicate_fun(id2) for id2 in curr_inc_ids])
            exclusion_str = ','.join([self.predicate_fun(id2) for id2 in curr_exc_ids])
            id1 = f"ex_{ex_num}_{i}"
            examples.append(self._str_example(id1, inclusion_str, exclusion_str, context_str, is_positive=True))

        context_str += '\n'
        # Negative example only says that all not included examples are negative
        for inc_ids in self.inclusions:
            inc_ids = set(inc_ids)
            exc_ids = set(range(0, self.max_concept_id)) - inc_ids

            pos_atoms = [self.predicate_fun(id) for id in inc_ids]
            neg_atoms = [f"not {self.predicate_fun(id)}" for id in exc_ids]
            body = ', '.join(pos_atoms + neg_atoms)
            context_str += f"goal :- {body}.\n"

        eg_id = f"ex_{ex_num}_{len(self.inclusions)}"
        examples.append(
            self._str_example(eg_id, inclusion_str="", exclusion_str="goal", context_str=context_str,
                              is_positive=False))

        return '\n\n'.join(examples)

    @staticmethod
    def _str_example(eg_id, inclusion_str, exclusion_str, context_str, is_positive):
        prefix = "pos" if is_positive else "neg"
        return f"#{prefix}({eg_id}@1,\n" \
               f"{{ {inclusion_str} }},\n" \
               f"{{ {exclusion_str} }}, \n" \
               f"{{\n" \
               f"{context_str}\n" \
               "}).\n"


def predicate_token(tok_id, tok_text):
    return f"token(tok{tok_id}, {tok_text})."


def get_predicate_function(problem_type: ProblemType) -> Callable[[int], str]:
    if problem_type == ProblemType.ATOMISATION:
        return predicate_atomic_sent
    return predicate_generalised_sent


def predicate_generalised_sent(tok_id: int) -> str:
    return f"in_generalised_sent(tok{tok_id})"


def predicate_atomic_sent(tok_id: int) -> str:
    return f"in_atomic_sent(tok{tok_id})"


def predicate_sentence_start(tok_id):
    return f"start(tok{tok_id})."


def predicate_sentence_successor(tok1_id, tok2_id):
    return f"succ(tok{tok1_id}, tok{tok2_id})."


def predicate_doc_root(root_id):
    return f"root(tok{root_id})."


def predicate_doc_dependency(parent_id, child_id, dep):
    return f"dep({dep}, tok{parent_id}, tok{child_id})."


def predicate_pos_token(tok_id, pos):
    return f"part_of_speech(tok{tok_id}, {pos})."
