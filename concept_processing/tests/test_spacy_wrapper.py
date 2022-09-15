from unittest import TestCase

from concept_processing.nlp.spacy_wrapper import SpacyWrapper


class TestSpacyWrapper(TestCase):
    def test_sentence_split(self):
        given = "Hello world! Another hello world."
        expect = ["Hello world!", "Another hello world."]

        spacy = SpacyWrapper()
        doc = spacy(given)

        sents = [str(sent) for sent in doc.sentences()]
        self.assertEqual(sents, expect)
