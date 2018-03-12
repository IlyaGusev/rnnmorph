import unittest
import logging
import sys
import numpy as np

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from rnnmorph.predictor import RNNMorphPredictor
from rnnmorph.tag_genres import tag_files


class TestLSTMMorph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        cls.predictor = RNNMorphPredictor()

    def __asert_parse(self, parse, pos, normal_form, tag):
        self.assertEqual(parse.pos, pos)
        self.assertEqual(parse.normal_form, normal_form)
        self.assertEqual(parse.tag, tag)

    def test_sentence_analysis1(self):
        forms = self.predictor.predict_sentence_tags(["косил", "косой", "косой", "косой"])
        self.__asert_parse(forms[0], 'VERB', 'косить',
                           'Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act')

    def test_sentence_analysis2(self):
        forms = self.predictor.predict_sentence_tags(["мама", "мыла", "раму"])
        self.__asert_parse(forms[0], 'NOUN', 'мама', 'Case=Nom|Gender=Fem|Number=Sing')
        self.__asert_parse(forms[1], 'VERB', 'мыть',
                           'Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act')
        self.__asert_parse(forms[2], 'NOUN', 'рама', 'Case=Dat|Gender=Masc|Number=Sing')

    def test_sentences_analysis1(self):
        forms = self.predictor.predict_sentences_tags([["косил", "косой", "косой", "косой"], ["мама", "мыла", "раму"]])

        self.__asert_parse(forms[0][0], 'VERB', 'косить',
                           'Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act')

        self.__asert_parse(forms[1][0], 'NOUN', 'мама', 'Case=Nom|Gender=Fem|Number=Sing')
        self.__asert_parse(forms[1][1], 'VERB', 'мыть',
                           'Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act')
        self.__asert_parse(forms[1][2], 'NOUN', 'рама', 'Case=Dat|Gender=Masc|Number=Sing')

    def test_proba(self):
        forms = self.predictor.predict_sentence_tags_proba(["косил", "косой", "косой", "косой"])
        for word_proba in forms:
            self.assertEqual(len(word_proba), 252)
        indices = np.array([pair[0] for pair in forms[2]]).argsort()[-5:][::-1]
        variants = [forms[2][i][1].tag for i in indices]
        self.assertIn('Case=Nom|Degree=Pos|Gender=Masc|Number=Sing', variants)

    def test_genres_accuracy(self):
        quality = tag_files(self.predictor)
        self.assertGreater(quality['Lenta'].tag_accuracy, 95)
        self.assertGreater(quality['Lenta'].sentence_accuracy, 70)
        self.assertGreater(quality['VK'].tag_accuracy, 93)
        self.assertGreater(quality['VK'].sentence_accuracy, 65)
        self.assertGreater(quality['JZ'].tag_accuracy, 94)
        self.assertGreater(quality['JZ'].sentence_accuracy, 70)
