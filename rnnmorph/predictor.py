# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Предсказатель PoS-тегов.

from typing import List
from collections import defaultdict

import nltk
import numpy as np
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters

from rnnmorph.model import LSTMMorphoAnalysis
from rnnmorph.data_preparation.process_tag import convert_from_opencorpora_tag, process_gram_tag
from rnnmorph.data_preparation.word_form import WordFormOut
from rnnmorph.config import BuildModelConfig
from rnnmorph.settings import MODELS_PATHS


class Predictor:
    """
    Интерфейс POS-теггера.
    """
    def predict(self, words: List[str], include_all_forms: bool) -> List[WordFormOut]:
        """
        Предсказать теги для одного предложения.
        
        :param words: массив слов (знаки препинания - отдельные токены).
        :param include_all_forms: флаг, включающий все варианты разбора.
        :return: массив форм с леммами, тегами и оригинальными словами.
        """
        raise NotImplementedError()

    def predict_sentences(self, sentences: List[List[str]], batch_size: int,
                          include_all_forms: bool) -> List[List[WordFormOut]]:
        """
        Предсказать теги для массива предложений. В сетку как batch загружается.
        
        :param sentences: массив предложений.
        :param batch_size: размер батча.
        :param include_all_forms: флаг, включающий все варианты разбора.
        :return: массив форм с леммами, тегами и оригинальными словами для каждого предложения.
        """
        raise NotImplementedError()


class RNNMorphPredictor(Predictor):
    """
    POS-теггер на освное RNN.
    """
    def __init__(self,
                 language="ru",
                 eval_model_config_path: str=None,
                 eval_model_weights_path: str=None,
                 gram_dict_input: str=None,
                 gram_dict_output: str=None,
                 word_vocabulary: str=None,
                 char_set_path: str=None,
                 build_config: str=None):
        if eval_model_config_path is None:
            eval_model_config_path = MODELS_PATHS[language]["eval_model_config"]
        if eval_model_weights_path is None:
            eval_model_weights_path = MODELS_PATHS[language]["eval_model_weights"]
        if gram_dict_input is None:
            gram_dict_input = MODELS_PATHS[language]["gram_input"]
        if gram_dict_output is None:
            gram_dict_output = MODELS_PATHS[language]["gram_output"]
        if word_vocabulary is None:
            word_vocabulary = MODELS_PATHS[language]["word_vocabulary"]
        if char_set_path is None:
            char_set_path = MODELS_PATHS[language]["char_set"]
        if build_config is None:
            build_config = MODELS_PATHS[language]["build_config"]

        self.language = language
        self.converter = converters.converter('opencorpora-int', 'ud14') if language == "ru" else None
        self.morph = MorphAnalyzer() if language == "ru" else None
        if self.language == "en":
            nltk.download("wordnet")
            nltk.download('averaged_perceptron_tagger')
            nltk.download('universal_tagset')

        self.build_config = BuildModelConfig()
        self.build_config.load(build_config)

        self.model = LSTMMorphoAnalysis(language=language)
        self.model.prepare(gram_dict_input, gram_dict_output, word_vocabulary, char_set_path)
        self.model.load_eval(self.build_config, eval_model_config_path, eval_model_weights_path)

    def predict(self, words: List[str], include_all_forms: bool=False) -> List[WordFormOut]:
        words_probabilities = self.model.predict_probabilities([words], 1, self.build_config)[0]
        return self.__get_sentence_forms(words, words_probabilities, include_all_forms)

    def predict_sentences(self, sentences: List[List[str]], batch_size: int=64,
                          include_all_forms: bool=False) -> List[List[WordFormOut]]:
        sentences_probabilities = self.model.predict_probabilities(sentences, batch_size, self.build_config)
        answers = []
        for words, words_probabilities in zip(sentences, sentences_probabilities):
            answers.append(self.__get_sentence_forms(words, words_probabilities, include_all_forms))
        return answers

    def __get_sentence_forms(self, words: List[str], words_probabilities: List[List[float]],
                             include_all_forms: bool) -> List[WordFormOut]:
        """
        Получить теги и формы.
        
        :param words: слова.
        :param words_probabilities: вероятности тегов слов.
        :param include_all_forms: флаг, включающий все варианты разбора.
        :return: вероятности и формы для всех вариантов слов.
        """
        result = []
        for word, word_prob in zip(words, words_probabilities[-len(words):]):
            result.append(self.__compose_out_form(word, word_prob[1:], include_all_forms))
        return result

    def __compose_out_form(self, word: str, probabilities: List[float],
                           include_all_forms: bool) -> WordFormOut:
        """
        Собрать форму по номеру теги в векторизаторе и слову.

        :param word: слово.
        :param probabilities: вероятности разных форм.
        :param include_all_forms: флаг, включающий все варианты разбора.
        :return: форма.
        """
        word_forms = None
        if self.language == "ru":
            word_forms = self.morph.parse(word)

        vectorizer = self.model.grammeme_vectorizer_output
        tag_num = int(np.argmax(probabilities))
        score = probabilities[tag_num]
        full_tag = vectorizer.get_name_by_index(tag_num)
        pos, tag = full_tag.split("#")[0], full_tag.split("#")[1]
        lemma = self.__get_lemma(word, pos, tag, word_forms)
        vector = np.array(vectorizer.get_vector(full_tag))
        result_form = WordFormOut(word=word, normal_form=lemma, pos=pos, tag=tag, vector=vector, score=score)

        if include_all_forms:
            weighted_vector = np.zeros_like(vector, dtype='float64')
            for tag_num, prob in enumerate(probabilities):
                full_tag = vectorizer.get_name_by_index(tag_num)
                pos, tag = full_tag.split("#")[0], full_tag.split("#")[1]
                lemma = self.__get_lemma(word, pos, tag, word_forms)
                vector = np.array(vectorizer.get_vector(full_tag), dtype='float64')
                weighted_vector += vector * prob

                form = WordFormOut(word=word, normal_form=lemma, pos=pos, tag=tag, vector=vector, score=prob)
                result_form.possible_forms.append(form)

            result_form.weighted_vector = weighted_vector
        return result_form

    def __get_lemma(self, word: str, pos_tag: str, gram: str, word_forms=None,
                    enable_normalization: bool=True):
        """
        Получить лемму.
        
        :param word: слово.
        :param pos_tag: часть речи.
        :param gram: граммаическое значение.
        :param enable_normalization: использовать ли нормализацию как в корпусе ГИКРЯ.
        :return: лемма.
        """
        if '_' in word:
            return word
        if self.language == "ru":
            if word_forms is None:
                word_forms = self.morph.parse(word)
            guess = ""
            max_common_tags = 0
            for word_form in word_forms:
                word_form_pos_tag, word_form_gram = convert_from_opencorpora_tag(self.converter, word_form.tag, word)
                word_form_gram = process_gram_tag(word_form_gram)
                common_tags_len = len(set(word_form_gram.split("|")).intersection(set(gram.split("|"))))
                if common_tags_len > max_common_tags and word_form_pos_tag == pos_tag:
                    max_common_tags = common_tags_len
                    guess = word_form
            if guess == "":
                guess = word_forms[0]
            if enable_normalization:
                lemma = self.__normalize_for_gikrya(guess)
            else:
                lemma = guess.normal_form
            return lemma
        elif self.language == "en":
            lemmatizer = nltk.stem.WordNetLemmatizer()
            pos_map = defaultdict(lambda: 'n')
            pos_map.update({
                'ADJ': 'a',
                'ADV': 'r',
                'NOUN': 'n',
                'VERB': 'v'
            })
            return lemmatizer.lemmatize(word, pos=pos_map[pos_tag])
        else:
            assert False

    @staticmethod
    def __normalize_for_gikrya(form):
        """
        Поучение леммы по правилам, максимально близким к тем, которые в корпусе ГИКРЯ.
        
        :param form: форма из pymorphy2.
        :return: леммма.
        """
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
