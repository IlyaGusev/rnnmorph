# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль загрузки корпусов.

from typing import List

import nltk
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters

from rnnmorph.data_preparation.grammeme_vectorizer import GrammemeVectorizer
from rnnmorph.data_preparation.word_vocabulary import WordVocabulary
from rnnmorph.util.tqdm_open import tqdm_open
from rnnmorph.data_preparation.process_tag import convert_from_opencorpora_tag, process_gram_tag


class Loader(object):
    """
    Класс для построения GrammemeVectorizer и WordVocabulary по корпусу
    """
    def __init__(self, language: str):
        self.language = language
        self.grammeme_vectorizer_input = GrammemeVectorizer()  # type: GrammemeVectorizer
        self.grammeme_vectorizer_output = GrammemeVectorizer()  # type: GrammemeVectorizer
        self.word_vocabulary = WordVocabulary()  # type: WordVocabulary
        self.char_set = set()
        self.morph = MorphAnalyzer() if self.language == "ru" else None  # type: MorphAnalyzer
        self.converter = converters.converter('opencorpora-int', 'ud14') if self.language == "ru" else None

    def parse_corpora(self, file_names: List[str]):
        """
        Построить WordVocabulary, GrammemeVectorizer по корпусу

        :param file_names: пути к файлам корпуса.
        """
        for file_name in file_names:
            with tqdm_open(file_name, encoding="utf-8") as f:
                for line in f:
                    if line == "\n":
                        continue
                    self.__process_line(line)

        self.grammeme_vectorizer_input.init_possible_vectors()
        self.grammeme_vectorizer_output.init_possible_vectors()
        self.word_vocabulary.sort()
        self.char_set = " " + "".join(self.char_set).replace(" ", "")

    def __process_line(self, line: str):
        """
        Обработка строчки в корпусе с морфоразметкой.
        :param line: 
        :return: 
        """
        text, lemma, pos_tag, grammemes = line.strip().split("\t")[0:4]
        # Заполняем словарь.
        self.word_vocabulary.add_word(text.lower())
        # Заполняем набор символов
        self.char_set |= {ch for ch in text}
        # Заполняем набор возможных выходных тегов.
        self.grammeme_vectorizer_output.add_grammemes(pos_tag, grammemes)
        # Заполняем набор возможных входных тегов.
        if self.language == "ru":
            for parse in self.morph.parse(text):
                pos, gram = convert_from_opencorpora_tag(self.converter, parse.tag, text)
                gram = process_gram_tag(gram)
                self.grammeme_vectorizer_input.add_grammemes(pos, gram)
        elif self.language == "en":
            _, tags = zip(*nltk.pos_tag([text], tagset='universal'))
            pos = tags[0]
            self.grammeme_vectorizer_input.add_grammemes(pos, "_")
