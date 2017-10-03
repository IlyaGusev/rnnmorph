# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Генератор батчей с определёнными параметрами.

from typing import List, Tuple

import pymorphy2
import numpy as np
from russian_tagsets import converters

from rnnmorph.data_preparation.grammeme_vectorizer import GrammemeVectorizer
from rnnmorph.data_preparation.process_tag import convert_from_opencorpora_tag, process_gram_tag
from rnnmorph.data_preparation.word_form import WordForm
from rnnmorph.util.tqdm_open import tqdm_open


CHAR_SET = " абвгдеёжзийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ" \
           "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-'\""


class BatchGenerator:
    """
    Генератор наборов примеров для обучения.
    """

    def __init__(self, filenames: List[str], batch_size: int, grammeme_vectorizer_input: GrammemeVectorizer,
                 grammeme_vectorizer_output: GrammemeVectorizer, sentence_len_low: int, sentence_len_high: int,
                 max_word_len: int, indices: np.array):
        self.filenames = filenames  # type: List[str]
        # Праметры батча.
        self.batch_size = batch_size  # type: int
        self.sentence_len_low = sentence_len_low  # type: int
        self.sentence_len_high = sentence_len_high  # type: int
        self.max_word_len = max_word_len  # type: int
        # Разбиение на выборки.
        self.indices = indices  # type: np.array
        # Подготовленные словари.
        self.grammeme_vectorizer_input = grammeme_vectorizer_input  # type: GrammemeVectorizer
        self.grammeme_vectorizer_output = grammeme_vectorizer_output  # type: GrammemeVectorizer
        self.morph = pymorphy2.MorphAnalyzer()  # type: pymorphy2.MorphAnalyzer

    def __to_tensor(self, sentences: List[List[WordForm]]) -> Tuple[np.array, np.array, np.array]:
        """
        Преобразование предложений в признаки и ответы.
        
        :param sentences: предложения (с разобранными словоформами).
        :return: индексы слов, грамматические векторы, индексы символов, ответы для всех предложений.
        """
        n = len(sentences)
        grammemes_count = self.grammeme_vectorizer_input.grammemes_count()

        grammemes = np.zeros((n, self.sentence_len_high, grammemes_count), dtype=np.float)
        chars = np.zeros((n, self.sentence_len_high, self.max_word_len), dtype=np.int)
        y = np.zeros((n, self.sentence_len_high), dtype=np.int)

        for i, sentence in enumerate(sentences):
            gram_vectors, char_vectors = \
                self.get_sample([x.text for x in sentence], self.morph,
                                self.grammeme_vectorizer_input, self.max_word_len)
            assert len(gram_vectors) == len(sentence) and \
                   len(char_vectors) == len(sentence)

            grammemes[i, -len(sentence):] = gram_vectors
            chars[i, -len(sentence):] = char_vectors
            y[i, -len(sentence):] = [word.gram_vector_index + 1 for word in sentence]
        y = y.reshape(y.shape[0], y.shape[1], 1)
        return grammemes, chars,  y

    @staticmethod
    def get_sample(sentence: List[str], morph: pymorphy2.MorphAnalyzer,
                   grammeme_vectorizer: GrammemeVectorizer, max_word_len: int):
        """
        Получние признаков для отдельного предложения.
        
        :param sentence: предложение.
        :param morph: морфология.
        :param grammeme_vectorizer: грамматический словарь. 
        :param max_word_len: количество обрабатываемых букв в слове.
        :return: индексы слов, грамматические векторы, индексы символов.
        """
        to_ud = converters.converter('opencorpora-int', 'ud14')
        word_char_vectors = []
        word_gram_vectors = []
        for word in sentence:
            char_indices = np.zeros(max_word_len)
            gram_value_indices = np.zeros(grammeme_vectorizer.grammemes_count())

            # Индексы символов слова.
            word_char_indices = [CHAR_SET.index(ch) if ch in CHAR_SET else len(CHAR_SET) for ch in word][:max_word_len]
            char_indices[-min(len(word), max_word_len):] = word_char_indices
            word_char_vectors.append(char_indices)

            # Грамматический вектор слова.
            # Складываем все возможные варианты разбора поэлементно.
            for parse in morph.parse(word):
                pos, gram = convert_from_opencorpora_tag(to_ud, parse.tag, word)
                gram = process_gram_tag(gram)
                gram_value_indices += np.array(grammeme_vectorizer.get_vector(pos + "#" + gram))
            # Нормируем по каждой категории отдельно.
            sorted_grammemes = sorted(grammeme_vectorizer.all_grammemes.items(), key=lambda x: x[0])
            index = 0
            for category, values in sorted_grammemes:
                mask = gram_value_indices[index:index+len(values)]
                s = sum(mask)
                gram_value_indices[index:index+len(values)] = mask/s
                index += len(values)
            word_gram_vectors.append(gram_value_indices)

        return word_gram_vectors, word_char_vectors

    def __iter__(self):
        """
        Получение очередного батча.

        :return: индексы словоформ, грамматические векторы, ответы-индексы.
        """
        sentences = [[]]
        i = 0
        for filename in self.filenames:
            with tqdm_open(filename, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        last_sentence = sentences[-1]
                        is_wrong_sentence = (i not in self.indices) or \
                                            (len(last_sentence) < self.sentence_len_low) or \
                                            (len(last_sentence) > self.sentence_len_high)
                        if is_wrong_sentence:
                            sentences.pop()
                        if len(sentences) >= self.batch_size:
                            yield self.__to_tensor(sentences)
                            sentences = []
                        sentences.append([])
                        i += 1
                    else:
                        word, lemma, pos, tags = line.split('\t')[0:4]
                        word, lemma = word.lower(), lemma.lower() + '_' + pos
                        gram_vector_index = self.grammeme_vectorizer_output.get_index_by_name(pos + "#" + tags)
                        sentences[-1].append(WordForm(lemma, gram_vector_index, word))
        if len(sentences[-1]) == 0:
            sentences.pop()
        yield self.__to_tensor(sentences)
