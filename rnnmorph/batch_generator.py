# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Генератор батчей с определёнными параметрами.

from typing import List, Tuple
from collections import namedtuple

import nltk
import numpy as np
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters

from rnnmorph.data_preparation.word_vocabulary import WordVocabulary
from rnnmorph.data_preparation.grammeme_vectorizer import GrammemeVectorizer
from rnnmorph.data_preparation.process_tag import convert_from_opencorpora_tag, process_gram_tag
from rnnmorph.util.tqdm_open import tqdm_open
from rnnmorph.config import TrainConfig, BuildModelConfig


WordForm = namedtuple("WordForm", "text gram_vector_index")


class BatchGenerator:
    """
    Генератор наборов примеров для обучения.
    """

    def __init__(self, language: str,
                 file_names: List[str],
                 config: TrainConfig,
                 grammeme_vectorizer_input: GrammemeVectorizer,
                 grammeme_vectorizer_output: GrammemeVectorizer,
                 indices: np.array,
                 word_vocabulary: WordVocabulary,
                 char_set: str,
                 build_config: BuildModelConfig):
        self.language = language
        self.file_names = file_names  # type: List[str]
        # Параметры батчей.
        self.batch_size = config.external_batch_size  # type: int
        self.bucket_borders = config.sentence_len_groups  # type: List[Tuple[int]]
        self.buckets = [list() for _ in range(len(self.bucket_borders))]
        self.build_config = build_config
        self.word_vocabulary = word_vocabulary
        self.char_set = char_set
        # Разбиение на выборки.
        self.indices = indices  # type: np.array
        # Подготовленные словари.
        self.grammeme_vectorizer_input = grammeme_vectorizer_input  # type: GrammemeVectorizer
        self.grammeme_vectorizer_output = grammeme_vectorizer_output  # type: GrammemeVectorizer
        self.morph = MorphAnalyzer() if self.language == "ru" else None  # type: MorphAnalyzer
        self.converter = converters.converter('opencorpora-int', 'ud14') if self.language == "ru" else None

    def __to_tensor(self, sentences: List[List[WordForm]]) -> Tuple[List, List]:
        """
        Преобразование предложений в признаки и ответы.

        :param sentences: предложения (с разобранными словоформами).
        :return: индексы слов, грамматические векторы, индексы символов, ответы для всех предложений.
        """
        n = len(sentences)
        grammemes_count = self.grammeme_vectorizer_input.grammemes_count()
        sentence_max_len = max([len(sentence) for sentence in sentences])

        data = []
        target = []

        words = np.zeros((n,  sentence_max_len), dtype=np.int)
        grammemes = np.zeros((n, sentence_max_len, grammemes_count), dtype=np.float)
        chars = np.zeros((n, sentence_max_len, self.build_config.char_max_word_length), dtype=np.int)
        y = np.zeros((n, sentence_max_len), dtype=np.int)

        for i, sentence in enumerate(sentences):
            word_indices, gram_vectors, char_vectors = self.get_sample(
                [x.text for x in sentence],
                language=self.language,
                converter=self.converter,
                morph=self.morph,
                grammeme_vectorizer=self.grammeme_vectorizer_input,
                max_word_len=self.build_config.char_max_word_length,
                word_vocabulary=self.word_vocabulary,
                word_count=self.build_config.word_max_count,
                char_set=self.char_set)
            assert len(word_indices) == len(sentence) and \
                   len(gram_vectors) == len(sentence) and \
                   len(char_vectors) == len(sentence)

            words[i, -len(sentence):] = word_indices
            grammemes[i, -len(sentence):] = gram_vectors
            chars[i, -len(sentence):] = char_vectors
            y[i, -len(sentence):] = [word.gram_vector_index + 1 for word in sentence]
        if self.build_config.use_word_embeddings:
            data.append(words)
        if self.build_config.use_gram:
            data.append(grammemes)
        if self.build_config.use_chars:
            data.append(chars)
        y = y.reshape(y.shape[0], y.shape[1], 1)
        target.append(y)
        if self.build_config.use_pos_lm:
            y_prev = np.zeros_like(y)
            y_prev[:, 1:] = y[:, :-1]
            target.append(y_prev.reshape(y.shape[0], y.shape[1], 1))
            y_next = np.zeros_like(y)
            y_next[:, :-1] = y[:, 1:]
            target.append(y_next.reshape(y.shape[0], y.shape[1], 1))
        if self.build_config.use_word_lm:
            words_prev = np.zeros_like(words)
            words_prev[:, 1:] = words[:, :-1]
            target.append(words_prev.reshape(words.shape[0], words.shape[1], 1))
            words_next = np.zeros_like(words)
            words_next[:, :-1] = words[:, 1:]
            target.append(words_next.reshape(words.shape[0], words.shape[1], 1))
        return data, target

    @staticmethod
    def get_sample(sentence: List[str],
                   language: str,
                   converter,
                   morph: MorphAnalyzer,
                   grammeme_vectorizer: GrammemeVectorizer,
                   max_word_len: int,
                   word_vocabulary: WordVocabulary,
                   word_count: int,
                   char_set: str):
        """
        Получние признаков для отдельного предложения.

        :param language: язык.
        :param sentence: предложение.
        :param morph: морфология.
        :param converter: конвертер тегов в UD.
        :param grammeme_vectorizer: грамматический словарь. 
        :param max_word_len: количество обрабатываемых букв в слове.
        :param word_vocabulary: список слов.
        :param word_count: максимальный индекс слова.
        :param char_set: список возможных символов, для которых есть эмбеддинги.
        :return: индексы слов, грамматические векторы, индексы символов.
        """
        word_char_vectors = []
        word_gram_vectors = []
        word_indices = []
        for word in sentence:
            char_indices = np.zeros(max_word_len)
            gram_value_indices = np.zeros(grammeme_vectorizer.grammemes_count())

            # Индексы символов слова.
            word_char_indices = [char_set.index(ch) if ch in char_set else len(char_set) for ch in word][-max_word_len:]
            char_indices[-min(len(word), max_word_len):] = word_char_indices
            word_char_vectors.append(char_indices)

            # Индексы слов.
            word_index = word_vocabulary.word_to_index[word.lower()] if word_vocabulary.has_word(word) else word_count
            word_index = min(word_index, word_count)
            word_indices.append(word_index)

            # Грамматический вектор слова.
            if language == "ru":
                # Складываем все возможные варианты разбора поэлементно.
                for parse in morph.parse(word):
                    pos, gram = convert_from_opencorpora_tag(converter, parse.tag, word)
                    gram = process_gram_tag(gram)
                    gram_value_indices += np.array(grammeme_vectorizer.get_vector(pos + "#" + gram))
            elif language == "en":
                _, tags = zip(*nltk.pos_tag([word], tagset='universal'))
                pos = tags[0]
                gram_value_indices += np.array(grammeme_vectorizer.get_vector(pos + "#_"))

            # Нормируем по каждой категории отдельно.
            sorted_grammemes = sorted(grammeme_vectorizer.all_grammemes.items(), key=lambda x: x[0])
            index = 0
            for category, values in sorted_grammemes:
                mask = gram_value_indices[index:index + len(values)]
                s = sum(mask)
                gram_value_indices[index:index + len(values)] = mask / s
                index += len(values)
            word_gram_vectors.append(gram_value_indices)

        return word_indices, word_gram_vectors, word_char_vectors

    def __iter__(self):
        """
        Получение очередного батча.

        :return: индексы словоформ, грамматические векторы, ответы-индексы.
        """
        last_sentence = []
        i = 0
        for filename in self.file_names:
            with tqdm_open(filename, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        if i not in self.indices:
                            last_sentence = []
                            i += 1
                            continue
                        for index, bucket in enumerate(self.buckets):
                            if self.bucket_borders[index][0] <= len(last_sentence) < self.bucket_borders[index][1]:
                                bucket.append(last_sentence)
                            if len(bucket) >= self.batch_size:
                                yield self.__to_tensor(bucket)
                                self.buckets[index] = []
                        last_sentence = []
                        i += 1
                    else:
                        word, _, pos, tags = line.split('\t')[0:4]
                        gram_vector_index = self.grammeme_vectorizer_output.get_index_by_name(pos + "#" + tags)
                        last_sentence.append(WordForm(text=word, gram_vector_index=gram_vector_index))
        for index, bucket in enumerate(self.buckets):
            yield self.__to_tensor(bucket)
