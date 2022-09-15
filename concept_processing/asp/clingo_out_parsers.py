import os.path
import re

from typing import List

from concept_processing.enums import ProblemType


# Generates sentences from the output of a clingo output
class ClingoAnsParser:
    def __init__(self, problem_type: ProblemType):
        problem_to_predicate_id = {
            ProblemType.ATOMISATION: "in_atomic_sent",
            ProblemType.GENERALISATION: "in_generalised_sent",
        }
        self.predicate_id = problem_to_predicate_id[problem_type]

    def get_sentences(self, clingo_out: List[List[str]]) -> List[str]:
        sents = []
        for atoms in clingo_out:
            words = self._create_id_words_mapping(atoms)
            generalised_tok_ids = self._extract_token_ids_from(self.predicate_id, atoms)

            sent = ' '.join([words[tok_id] for tok_id in generalised_tok_ids])
            sents.append(sent)
        return sents

    # Assuming file contains #show dep_chain/2.
    # Gets dep_chain values which are used to reduce ILASP search space size
    def get_all_dep_chains(self, clingo_out: List[List[str]]) -> List[str]:
        sol = set()
        for line in clingo_out:
            sol |= set(line)
        return list(sol)

    @staticmethod
    def _extract_token_ids_from(predicate, atoms):
        # match in_generalised_sent(tok2)
        match_generalised_sent_tok = list(map(lambda s: re.match(rf'{predicate}\(tok(\d+)\)', s),
                                              atoms))
        match_generalised_sent_tok = list(filter(None, match_generalised_sent_tok))
        tok_id_pos = 1
        generalised_tok_ids = [int(r.group(tok_id_pos)) for r in match_generalised_sent_tok]
        generalised_tok_ids.sort()
        return generalised_tok_ids

    @staticmethod
    def _create_id_words_mapping(atoms):
        # match token(tok3, "2nd")
        match_token_atoms = map(lambda s: re.match(rf'token\(tok(\d+),\"(.+)\"\)', s),
                                atoms)
        match_token_atoms = filter(None, match_token_atoms)
        tok_id_pos, tok_word_pos = 1, 2
        token_tuples = {int(r.group(tok_id_pos)): r.group(tok_word_pos) for r in match_token_atoms}
        return token_tuples


# Parses clingo examples to determine which original sentence do they correspond to
# Used for evaluation
class ClingoExParser:
    def __init__(self, outfile: str):
        self.outfile = outfile

    def get_row_ids(self) -> List[int]:
        if not os.path.exists(self.outfile):
            return []

        with open(self.outfile, 'r') as f:
            matches = filter(None, map(lambda s: re.match(rf'#(pos|neg)\(ex_(\d+)_(\d+)@(\d+),', s), f))
            row_ids_pos = 2

            return list({int(match.group(row_ids_pos)) for match in matches})
