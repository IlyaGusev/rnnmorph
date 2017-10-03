# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Предсказатель PoS-тегов.

from typing import List
from collections import namedtuple

from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters

from rnnmorph.model import LSTMMorphoAnalysis
from rnnmorph.data_preparation.process_tag import convert_from_opencorpora_tag, process_gram_tag


class MorphPredictor:
    def __init__(self, model_config_path: str, model_weights_path: str,
                 gramm_dict_input: str, gramm_dict_output: str):
        self.model = LSTMMorphoAnalysis()
        self.model.prepare(gramm_dict_input, gramm_dict_output)
        self.model.load(model_config_path, model_weights_path)
        self.morph = MorphAnalyzer()

    def predict(self, words: List[str]) -> List[str]:
        Parse = namedtuple('Parse', "word pos tag normal_form")
        tags = self.model.predict(words)
        forms = []
        for i, word in enumerate(words):
            pos_tag = tags[i].split("#")[0]
            gram = tags[i].split("#")[1]
            lemma = self.__get_lemma(word, pos_tag, gram)
            forms.append(Parse(word=word, pos=pos_tag, tag=gram, normal_form=lemma))
        return forms

    def __get_lemma(self, word: str, pos_tag: str, gram: str, enable_gikrya_normalization: bool=True):
        if '_' in word:
            return word
        to_ud = converters.converter('opencorpora-int', 'ud14')
        guess = ""
        max_common_tags = 0
        for word_form in self.morph.parse(word):
            word_form_pos_tag, word_form_gram = convert_from_opencorpora_tag(to_ud, word_form.tag, word)
            word_form_gram = process_gram_tag(word_form_gram)
            common_tags_len = len(set(word_form_gram.split("|")).intersection(set(gram.split("|"))))
            if common_tags_len > max_common_tags and word_form_pos_tag == pos_tag:
                max_common_tags = common_tags_len
                guess = word_form
        if guess == "":
            guess = self.morph.parse(word)[0]
        if enable_gikrya_normalization:
            lemma = self.__normalize_for_gikrya(guess)
        else:
            lemma = guess.normal_form
        return lemma

    @staticmethod
    def __normalize_for_gikrya(form):
        if form.tag.POS == 'NPRO':
            if form.normal_form == 'она':
                return 'он'
            if form.normal_form == 'они':
                return 'он'
            if form.normal_form == 'оно':
                return 'он'

        if form.word == 'об':
            return 'об'
        if form.word == 'тот':
            return 'то'
        if form.word == 'со':
            return 'со'

        if form.tag.POS in {'PRTS', 'PRTF'}:
            return form.inflect({'PRTF', 'sing', 'masc', 'nomn'}).word

        return form.normal_form
