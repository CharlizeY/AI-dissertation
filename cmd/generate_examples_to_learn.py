"""
Generated postive examples ILASP can learn from
"""

import concept_processing.io
from concept_processing.asp.batch_asp_generator import BatchASPGenerator
from concept_processing.enums import ProblemType
from concept_processing.nlp.spacy_wrapper import SpacyWrapper

problem_type = ProblemType.GENERALISATION

base_dir = {
    ProblemType.ATOMISATION: '../ilasp/atomisation',
    ProblemType.GENERALISATION: '../ilasp/generalisation',
}

concepts_file_dict = {
    ProblemType.ATOMISATION: f'{base_dir[ProblemType.ATOMISATION]}/created_examples/examples.csv',
    ProblemType.GENERALISATION: f'{base_dir[ProblemType.GENERALISATION]}/created_examples'
                                f'/examples.csv',
}

outfile_template = f'{base_dir[problem_type]}/ilasp_examples/examples_{{}}.ilasp'

if __name__ == '__main__':
    concepts_file = concepts_file_dict[problem_type]

    p_texts, c_texts = concept_processing.io.load_concept_examples(concepts_file)

    batch_asp_generator = BatchASPGenerator(SpacyWrapper(), problem_type)
    batch_asp_generator.parse(p_texts, c_texts)
    examples = batch_asp_generator.get_examples(n_folds=10)
    for i, ex in enumerate(examples):
        with open(outfile_template.format(i), 'w') as o:
            o.write(ex)
