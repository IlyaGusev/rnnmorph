import unittest
from rnnmorph.predictor import RNNMorphPredictor
from rnnmorph.settings import RU_MORPH_DEFAULT_MODEL_CONFIG, RU_MORPH_DEFAULT_MODEL_WEIGHTS, \
    RU_MORPH_GRAMMEMES_DICT, RU_MORPH_GRAMMEMES_DICT_OUTPUT


class TestLSTMMorph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = RNNMorphPredictor(RU_MORPH_DEFAULT_MODEL_CONFIG, RU_MORPH_DEFAULT_MODEL_WEIGHTS,
                                          RU_MORPH_GRAMMEMES_DICT, RU_MORPH_GRAMMEMES_DICT_OUTPUT)

    def __asert_parse(self, parse, pos, normal_form, tag):
        self.assertEqual(parse.pos, pos)
        self.assertEqual(parse.normal_form, normal_form)
        self.assertEqual(parse.tag, tag)

    def test_sentence_analysis1(self):
        forms = self.predictor.predict_sentence_tags(["косил", "косой", "косой", "косой"])
        self.__asert_parse(forms[0], 'VERB', 'косить',
                           'Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act')
        self.__asert_parse(forms[3], 'NOUN', 'коса', 'Case=Ins|Gender=Fem|Number=Sing')

    def test_sentence_analysis2(self):
        forms = self.predictor.predict_sentence_tags(["мама", "мыла", "раму"])
        self.__asert_parse(forms[0], 'NOUN', 'мама', 'Case=Nom|Gender=Fem|Number=Sing')
        self.__asert_parse(forms[1], 'VERB', 'мыть',
                           'Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act')
        self.__asert_parse(forms[2], 'NOUN', 'рама', 'Case=Dat|Gender=Masc|Number=Sing')