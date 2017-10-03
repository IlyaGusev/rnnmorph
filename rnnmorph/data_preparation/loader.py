# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль загрузки корпусов.

from typing import List, Tuple

import pymorphy2
from russian_tagsets import converters

from rnnmorph.data_preparation.grammeme_vectorizer import GrammemeVectorizer
from rnnmorph.util.tqdm_open import tqdm_open
from rnnmorph.data_preparation.process_tag import convert_from_opencorpora_tag, process_gram_tag


class Loader(object):
    """
    Класс для построения GrammemeVectorizer и WordVocabulary по корпусу
    """
    def __init__(self, gram_dump_path_input: str, gram_dump_path_output: str):
        self.grammeme_vectorizer_input = GrammemeVectorizer(gram_dump_path_input)  # type: GrammemeVectorizer
        self.grammeme_vectorizer_output = GrammemeVectorizer(gram_dump_path_output)  # type: GrammemeVectorizer
        self.morph = pymorphy2.MorphAnalyzer()  # type: pymorphy2.MorphAnalyzer
        self.converter = converters.converter('opencorpora-int', 'ud14')

    def parse_corpora(self, filenames: List[str]) -> Tuple[GrammemeVectorizer, GrammemeVectorizer]:
        """
        Построить WordVocabulary, GrammemeVectorizer по корпусу

        :param filenames: пути к файлам корпуса.
        """
        for filename in filenames:
            with tqdm_open(filename, encoding="utf-8") as f:
                for line in f:
                    if line == "\n":
                        continue
                    self.__process_line(line)

        self.grammeme_vectorizer_input.init_possible_vectors()
        self.grammeme_vectorizer_output.init_possible_vectors()
        return self.grammeme_vectorizer_input, self.grammeme_vectorizer_output

    def __process_line(self, line: str) -> None:
        """
        Обработка строчки в корпусе с морфоразметкой.
        :param line: 
        :return: 
        """
        text, lemma, pos_tag, grammemes = line.strip().split("\t")[0:4]
        # Заполняем набор возможных выходных тегов.
        self.grammeme_vectorizer_output.add_grammemes(pos_tag, grammemes)
        # Заполняем набор возможных входных тегов.
        for parse in self.morph.parse(text):
            pos, gram = convert_from_opencorpora_tag(self.converter, parse.tag, text)
            gram = process_gram_tag(gram)
            self.grammeme_vectorizer_input.add_grammemes(pos, gram)
