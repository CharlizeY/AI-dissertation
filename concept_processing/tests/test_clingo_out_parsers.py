from pyfakefs.fake_filesystem_unittest import TestCase

from concept_processing.asp.clingo_out_parsers import ClingoExParser


# TODO: fix pyfakefs doesn't work with test discovery
# class TestClingoExParser(TestCase):
#     def setUp(self) -> None:
#         print("Starting setup")
#         self.setUpPyfakefs()
#         print("Done setup")
#         self.file_path = 'test.txt'
#
#     def test_empty_list_for_empty_file(self):
#         file_contents = ''''''
#         expect = []
#
#         self.fs.create_file(self.file_path, contents=file_contents)
#
#         parser = ClingoExParser(self.file_path)
#         got = parser.get_row_ids()
#         self.assertEqual(expect, got)
#
#     def test_empty_list_for_not_existing_file(self):
#         expect = []
#
#         parser = ClingoExParser("")
#         got = parser.get_row_ids()
#         self.assertEqual(expect, got)
#
#     def test_works_for_1_digit_ids(self):
#         file_contents = '''
# #pos(ex_1_0@1,
# { in_generalised_sent(tok0),in_generalised_sent(tok1),in_generalised_sent(tok2),in_generalised_sent(tok3),in_generalised_sent(tok4),in_generalised_sent(tok8) },
# { in_generalised_sent(tok5),in_generalised_sent(tok6),in_generalised_sent(tok7) },
# {
# token(tok0, "the").
# token(tok1, "fielder").
# token(tok2, "caught").
# token(tok3, "the").
# token(tok4, "ball").
# token(tok5, "for").
# token(tok6, "an").
# token(tok7, "out").
# token(tok8, ".").
# start(tok0).
# succ(tok0, tok1).
# succ(tok1, tok2).
# succ(tok2, tok3).
# succ(tok3, tok4).
# succ(tok4, tok5).
# }).
# '''
#         expect = [1]
#
#         self.fs.create_file(self.file_path, contents=file_contents)
#         parser = ClingoExParser(self.file_path)
#         got = parser.get_row_ids()
#         self.assertEqual(expect, got)
#
#     def test_works_for_negative(self):
#         file_contents = '''
# #neg(ex_1_0@1,
# { in_generalised_sent(tok0),in_generalised_sent(tok1),in_generalised_sent(tok2),in_generalised_sent(tok3),in_generalised_sent(tok4),in_generalised_sent(tok8) },
# { in_generalised_sent(tok5),in_generalised_sent(tok6),in_generalised_sent(tok7) },
# {
# token(tok0, "the").
# token(tok1, "fielder").
# token(tok2, "caught").
# token(tok3, "the").
# token(tok4, "ball").
# token(tok5, "for").
# token(tok6, "an").
# token(tok7, "out").
# token(tok8, ".").
# start(tok0).
# succ(tok0, tok1).
# succ(tok1, tok2).
# succ(tok2, tok3).
# succ(tok3, tok4).
# succ(tok4, tok5).
# }).
#         '''
#         expect = [1]
#
#         self.fs.create_file(self.file_path, contents=file_contents)
#         parser = ClingoExParser(self.file_path)
#         got = parser.get_row_ids()
#         self.assertEqual(expect, got)
#
#     def test_works_for_5_digit_ids(self):
#         file_contents = '''
# #pos(ex_14854_0@1,
# { in_generalised_sent(tok0),in_generalised_sent(tok1),in_generalised_sent(tok2),in_generalised_sent(tok3),in_generalised_sent(tok4),in_generalised_sent(tok8) },
# { in_generalised_sent(tok5),in_generalised_sent(tok6),in_generalised_sent(tok7) },
# {
# token(tok0, "the").
# token(tok1, "fielder").
# token(tok2, "caught").
# token(tok3, "the").
# token(tok4, "ball").
# token(tok5, "for").
# token(tok6, "an").
# token(tok7, "out").
# token(tok8, ".").
# start(tok0).
# succ(tok0, tok1).
# succ(tok1, tok2).
# succ(tok2, tok3).
# succ(tok3, tok4).
# succ(tok4, tok5).
# }).
#         '''
#         expect = [14854]
#
#         self.fs.create_file(self.file_path, contents=file_contents)
#         parser = ClingoExParser(self.file_path)
#         got = parser.get_row_ids()
#         self.assertEqual(expect, got)
#
#     def test_multiple_ex(self):
#         file_contents = '''
# #neg(ex_14854_0@1,
# { in_generalised_sent(tok0),in_generalised_sent(tok1),in_generalised_sent(tok2),in_generalised_sent(tok3),in_generalised_sent(tok4),in_generalised_sent(tok8) },
# { in_generalised_sent(tok5),in_generalised_sent(tok6),in_generalised_sent(tok7) },
# {
# token(tok0, "the").
# token(tok1, "fielder").
# token(tok2, "caught").
# token(tok3, "the").
# token(tok4, "ball").
# token(tok5, "for").
# token(tok6, "an").
# token(tok7, "out").
# token(tok8, ".").
# start(tok0).
# succ(tok0, tok1).
# succ(tok1, tok2).
# succ(tok2, tok3).
# succ(tok3, tok4).
# succ(tok4, tok5).
# }).
#
#
# #pos(ex_1_0@1,
# { in_generalised_sent(tok0),in_generalised_sent(tok1),in_generalised_sent(tok2),in_generalised_sent(tok3),in_generalised_sent(tok4),in_generalised_sent(tok8) },
# { in_generalised_sent(tok5),in_generalised_sent(tok6),in_generalised_sent(tok7) },
# {
# token(tok0, "the").
# token(tok1, "fielder").
# token(tok2, "caught").
# token(tok3, "the").
# token(tok4, "ball").
# token(tok5, "for").
# token(tok6, "an").
# token(tok7, "out").
# token(tok8, ".").
# start(tok0).
# succ(tok0, tok1).
# succ(tok1, tok2).
# succ(tok2, tok3).
# succ(tok3, tok4).
# succ(tok4, tok5).
# }).
#         '''
#         expect = [14854, 1]
#
#         self.fs.create_file(self.file_path, contents=file_contents)
#         parser = ClingoExParser(self.file_path)
#         got = parser.get_row_ids()
#         # Can be in different order
#         self.assertEqual(set(expect), set(got))
#
#     def test_deduplicate(self):
#         file_contents = '''
# #neg(ex_1_0@1,
# { in_generalised_sent(tok0),in_generalised_sent(tok1),in_generalised_sent(tok2),in_generalised_sent(tok3),in_generalised_sent(tok4),in_generalised_sent(tok8) },
# { in_generalised_sent(tok5),in_generalised_sent(tok6),in_generalised_sent(tok7) },
# {
# token(tok0, "the").
# token(tok1, "fielder").
# token(tok2, "caught").
# token(tok3, "the").
# token(tok4, "ball").
# token(tok5, "for").
# token(tok6, "an").
# token(tok7, "out").
# token(tok8, ".").
# start(tok0).
# succ(tok0, tok1).
# succ(tok1, tok2).
# succ(tok2, tok3).
# succ(tok3, tok4).
# succ(tok4, tok5).
# }).
#
#
# #pos(ex_1_1@1,
# { in_generalised_sent(tok0),in_generalised_sent(tok1),in_generalised_sent(tok2),in_generalised_sent(tok3),in_generalised_sent(tok4),in_generalised_sent(tok8) },
# { in_generalised_sent(tok5),in_generalised_sent(tok6),in_generalised_sent(tok7) },
# {
# token(tok0, "the").
# token(tok1, "fielder").
# token(tok2, "caught").
# token(tok3, "the").
# token(tok4, "ball").
# token(tok5, "for").
# token(tok6, "an").
# token(tok7, "out").
# token(tok8, ".").
# start(tok0).
# succ(tok0, tok1).
# succ(tok1, tok2).
# succ(tok2, tok3).
# succ(tok3, tok4).
# succ(tok4, tok5).
# }).
#         '''
#         expect = [1]
#
#         self.fs.create_file(self.file_path, contents=file_contents)
#         parser = ClingoExParser(self.file_path)
#         got = parser.get_row_ids()
#         self.assertEqual(expect, got)
