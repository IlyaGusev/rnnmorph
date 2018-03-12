# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Предсказатель PoS-тегов.

from typing import List, Tuple

from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters

from rnnmorph.model import LSTMMorphoAnalysis
from rnnmorph.data_preparation.process_tag import convert_from_opencorpora_tag, process_gram_tag
from rnnmorph.data_preparation.word_form import WordForm, WordFormOut
from rnnmorph.settings import RU_MORPH_DEFAULT_MODEL_CONFIG, RU_MORPH_DEFAULT_MODEL_WEIGHTS, \
    RU_MORPH_GRAMMEMES_DICT, RU_MORPH_GRAMMEMES_DICT_OUTPUT


class Predictor:
    """
    Интерфейс POS-теггера.
    """
    def predict_sentence_tags(self, words: List[str]) -> List[WordFormOut]:
        """
        Предсказать теги для одного предложения.
        
        :param words: массив слов (знаки препинания - отдельные токены). 
        :return: массив форм с леммами, тегами и оригинальными словами.
        """
        raise NotImplementedError()

    def predict_sentences_tags(self, sentences: List[List[str]]) -> List[List[WordFormOut]]:
        """
        Предсказать теги для массива предложений. В сетку как batch загружается.
        
        :param sentences: массив предложений.
        :return: массив форм с леммами, тегами и оригинальными словами для каждого предложения.
        """
        raise NotImplementedError()

    def predict_sentence_tags_proba(self, words: List[str]) -> List[List[Tuple[float, WordFormOut]]]:
        """
        Предсказать вероятности тегов для слов в предложении.
        
        :param words: массив слов (знаки препинания - отдельные токены).
        :return: массив с вероятностями форм для каждого слова.
        """
        raise NotImplementedError()

    def predict_sentences_tags_proba(self, words: List[List[str]]) -> List[List[List[Tuple[float, WordFormOut]]]]:
        """
        Предсказать вероятности тегов для слов для массива предложений.

        :param words: массив предложений.
        :return: массив с вероятностями форм для каждого слова в каждом предложении.
        """
        raise NotImplementedError()


class RNNMorphPredictor(Predictor):
    """
    POS-теггер на освное RNN.
    """
    def __init__(self, model_config_path: str=RU_MORPH_DEFAULT_MODEL_CONFIG,
                 model_weights_path: str=RU_MORPH_DEFAULT_MODEL_WEIGHTS,
                 gramm_dict_input: str=RU_MORPH_GRAMMEMES_DICT,
                 gramm_dict_output: str=RU_MORPH_GRAMMEMES_DICT_OUTPUT):
        self.model = LSTMMorphoAnalysis()
        self.model.prepare(gramm_dict_input, gramm_dict_output)
        self.model.load(model_config_path, model_weights_path)
        self.morph = MorphAnalyzer()

    def predict_sentence_tags(self, words: List[str]) -> List[WordFormOut]:
        tags = self.model.predict([words], batch_size=1)[0]
        return [self.__compose_out_form(tag_num, word) for tag_num, word in zip(tags, words)]

    def predict_sentences_tags(self, sentences: List[List[str]], batch_size: int=64) -> List[List[WordFormOut]]:
        sentences_tags = self.model.predict(sentences, batch_size)
        answers = []
        for tags, words in zip(sentences_tags, sentences):
            answers.append([self.__compose_out_form(tag_num, word) for tag_num, word in zip(tags, words)])
        return answers

    def predict_sentence_tags_proba(self, words: List[str]) -> List[List[Tuple[float, WordFormOut]]]:
        words_probabilities = self.model.predict_proba([words], batch_size=1)[0]
        return self.__get_sentence_forms_probs(words, words_probabilities)

    def predict_sentences_tags_proba(self, sentences: List[List[str]],
                                     batch_size: int=64) -> List[List[List[Tuple[float, WordFormOut]]]]:
        result = []
        sentences_probabilities = self.model.predict_proba(sentences, batch_size)
        for sentence, words_probabilities in zip(sentences, sentences_probabilities):
            result.append(self.__get_sentence_forms_probs(sentence, words_probabilities))
        return result

    def __get_sentence_forms_probs(self, words: List[str], words_probabilities: List[List[float]]) -> \
            List[List[Tuple[float, WordFormOut]]]:
        """
        Получить теги и формы.
        
        :param words: слова.
        :param words_probabilities: вероятности тегов слов.
        :return: вероятности и формы для всех вариантов слов.
        """
        result = []
        for word, word_prob in zip(words, words_probabilities[-len(words):]):
            word_prob = word_prob[1:]
            word_forms = [(grammeme_prob, self.__compose_out_form(tag_num, word))
                          for tag_num, grammeme_prob in enumerate(word_prob)]
            result.append(word_forms)
        return result

    def __compose_out_form(self, tag_num: int, word: str) -> WordFormOut:
        """
        Собрать форму по номеру теги в векторизаторе и слову.
        
        :param tag_num: номер тега.
        :param word: слово.
        :return: форма.
        """
        vectorizer = self.model.grammeme_vectorizer_output
        tag = vectorizer.get_name_by_index(tag_num)
        pos_tag = tag.split("#")[0]
        gram = tag.split("#")[1]
        lemma = self.__get_lemma(word, pos_tag, gram)
        return WordForm(lemma=lemma, gram_vector_index=tag_num, text=word).get_out_form(vectorizer)

    def __get_lemma(self, word: str, pos_tag: str, gram: str, enable_gikrya_normalization: bool=True):
        """
        Получить лемму.
        
        :param word: слово.
        :param pos_tag: часть речи.
        :param gram: граммаическое значение.
        :param enable_gikrya_normalization: использовать ли нормализацию как в корпусе ГИКРЯ.
        :return: лемма.
        """
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
