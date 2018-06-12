# -*- coding: utf-8 -*-
# Авторы: Гусев Илья
# Описание: Словоформа.

import numpy as np


class WordFormOut(object):
    """
    Класс словоформы.
    """
    def __init__(self, word: str, normal_form: str, pos: str, tag: str, vector: np.array, score: float):
        """
        :param word: вокабула словоформы.
        :param normal_form: лемма словоформы (=начальная форма, нормальная форма).
        :param pos: часть речи.
        :param tag: грамматическое значение.
        :param vector: вектор словоформы.
        :param score: вероятность словоформы.
        """
        self.word = word
        self.normal_form = normal_form
        self.pos = pos
        self.tag = tag
        self.vector = vector
        self.score = score
        self.weighted_vector = np.zeros_like(self.vector)
        self.possible_forms = []

    def __repr__(self):
        return "<Normal form = {}; Word = {}>".format(self.normal_form, self.word)

    def __eq__(self, other):
        return (self.normal_form, self.word, self.pos, self.tag) == \
               (other.normal_form, other.word, other.pos, other.tag)

    def __hash__(self):
        return hash((self.normal_form, self.word, self.pos, self.tag))
