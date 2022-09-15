import unittest

from concept_processing.nlp.nlp_utils import truecase, merge_not
from concept_processing.nlp.spacy_wrapper import SpacyWrapper


class TestNLPUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.nlp = SpacyWrapper()

    def test_truecase_capitalises_sentence_start(self):
        given = "hello world to you too!"
        expect = "Hello world to you too!"
        got = truecase(given, self.nlp)
        self.assertEqual(expect, got, "Incorrect truecasing")

    def test_truecase_recognises_a_person(self):
        given = "hello john."
        expect = "Hello John."
        got = truecase(given, self.nlp)
        self.assertEqual(expect, got, "Incorrect truecasing")

    def test_truecase_works_with_multiple_sentences(self):
        given = "hello john. it is a nice day outside."
        expect = "Hello John. It is a nice day outside."
        got = truecase(given, self.nlp)
        self.assertEqual(expect, got, "Incorrect truecasing")

    def test_trucase_Roko_apostope_s_case(self):
        given = "he took ivan's bat"
        expect = "He took Ivan's bat"
        got = truecase(given, self.nlp)
        self.assertEqual(expect, got, "Incorrect truecasing")

    def test_should_merge_not(self):
        given = "The hitter have n't swing."
        expect = "The hitter haven't swing."

        got = merge_not(given)
        self.assertEqual(expect, got, "Should have merged have n't.")

    def test_should_not_merge_not(self):
        given = "The hitter have not swing."

        got = merge_not(given)
        self.assertEqual(given, got, "Should not have done anything")
