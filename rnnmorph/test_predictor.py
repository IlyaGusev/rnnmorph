import unittest
from rnnmorph.predictor import MorphPredictor
from rnnmorph.settings import RU_MORPH_DEFAULT_MODEL, RU_MORPH_WORD_VOCAB_DUMP, \
    RU_MORPH_GRAMMEMES_DICT, RU_MORPH_GRAMMEMES_DICT_OUTPUT


class TestLSTMMorph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = MorphPredictor(RU_MORPH_DEFAULT_MODEL, RU_MORPH_WORD_VOCAB_DUMP,
                                       RU_MORPH_GRAMMEMES_DICT, RU_MORPH_GRAMMEMES_DICT_OUTPUT)

    def test_sentence_analysis(self):
        self.assertEqual(self.predictor.predict(['один', 'жил', 'в', 'пустыне', 'рыбак', 'молодой']),
                         ['NUM#Case=Nom|Gender=Masc',
                          'VERB#Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act',
                          'ADP#_',
                          'NOUN#Case=Loc|Gender=Fem|Number=Sing',
                          'NOUN#Case=Nom|Gender=Masc|Number=Sing',
                          'ADJ#Case=Gen|Degree=Pos|Gender=Fem|Number=Sing'])