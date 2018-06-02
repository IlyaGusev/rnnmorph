# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль загрузки корпусов.

from typing import List, Tuple

import pymorphy2
from russian_tagsets import converters

from rnnmorph.data_preparation.grammeme_vectorizer import GrammemeVectorizer
from rnnmorph.data_preparation.word_vocabulary import WordVocabulary
from rnnmorph.util.tqdm_open import tqdm_open
from rnnmorph.data_preparation.process_tag import convert_from_opencorpora_tag, process_gram_tag


class Loader(object):
    """
    Класс для построения GrammemeVectorizer и WordVocabulary по корпусу
    """
    def __init__(self):
        self.grammeme_vectorizer_input = GrammemeVectorizer()  # type: GrammemeVectorizer
        self.grammeme_vectorizer_output = GrammemeVectorizer()  # type: GrammemeVectorizer
        self.word_vocabulary = WordVocabulary()  # type: WordVocabulary
        self.morph = pymorphy2.MorphAnalyzer()  # type: pymorphy2.MorphAnalyzer
        self.converter = converters.converter('opencorpora-int', 'ud14')

    def parse_corpora(self, file_names: List[str]) -> Tuple[GrammemeVectorizer, GrammemeVectorizer, WordVocabulary]:
        """
        Построить WordVocabulary, GrammemeVectorizer по корпусу

        :param file_names: пути к файлам корпуса.
        """
        for filename in file_names:
            with tqdm_open(filename, encoding="utf-8") as f:
                for line in f:
                    if line == "\n":
                        continue
                    self.__process_line(line)

        self.grammeme_vectorizer_input.init_possible_vectors()
        self.grammeme_vectorizer_output.init_possible_vectors()
        self.word_vocabulary.sort()
        return self.grammeme_vectorizer_input, self.grammeme_vectorizer_output, self.word_vocabulary

    def __process_line(self, line: str) -> None:
        """
        Обработка строчки в корпусе с морфоразметкой.
        :param line: 
        :return: 
        """
        text, lemma, pos_tag, grammemes = line.strip().split("\t")[0:4]
        # Заполняем словарь.
        self.word_vocabulary.add_word(text.lower())
        # Заполняем набор возможных выходных тегов.
        self.grammeme_vectorizer_output.add_grammemes(pos_tag, grammemes)
        # Заполняем набор возможных входных тегов.
        for parse in self.morph.parse(text):
            pos, gram = convert_from_opencorpora_tag(self.converter, parse.tag, text)
            gram = process_gram_tag(gram)
            self.grammeme_vectorizer_input.add_grammemes(pos, gram)
