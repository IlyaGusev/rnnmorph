# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модель PoS-теггера на основе BiLSTM.

from typing import List, Tuple

import numpy as np
import pymorphy2
from keras.layers import Input, Embedding, Dense, LSTM, BatchNormalization, Activation, \
    concatenate, Bidirectional, TimeDistributed, Dropout
from keras.models import Model, model_from_yaml
from keras.optimizers import Adam

from rnnmorph.batch_generator import BatchGenerator, CHAR_SET
from rnnmorph.data_preparation.grammeme_vectorizer import GrammemeVectorizer
from rnnmorph.data_preparation.loader import Loader


class LSTMMorphoAnalysis:
    def __init__(self, input_size: int=1000, external_batch_size: int=3000, nn_batch_size: int=256,
                 sentence_len_groups: Tuple=((1, 14), (15, 25), (26, 40), (40, 50)),
                 lstm_units=128, embeddings_dimension: int=150, dense_units: int=128, max_word_len: int=30,
                 char_embeddings_dimension: int=20, char_lstm_output_dim: int=64, dropout: float=0.2):
        """
        :param input_size: размер набора слова, которые кодируются эмбеддингами.
        :param external_batch_size: размер батча, который читается из файлов.
        :param nn_batch_size: размер батча для сети.
        :param sentence_len_groups: диапазоны количества слов в предложении.
        :param lstm_units: размер состояния у LSTM слоя. (у BiLSTM = lstm_units * 2).
        :param embeddings_dimension: размерность словных эмбеддингов.
        :param dense_units: размер выхода скрытого слоя.
        :param max_word_len: максимальная учитываемая длина слова.
        :param char_embeddings_dimension: размерность буквенных эмбеддингов.
        :param char_lstm_output_dim: размерность эмбеддинга слова, собранного на основе букевнных.
        :param dropout: дропаут на основных слоях.
        """
        self.external_batch_size = external_batch_size  # type: int
        self.sentence_len_groups = sentence_len_groups  # type: Tuple[Tuple[int, int]]
        self.nn_batch_size = nn_batch_size  # type: int
        self.max_word_len = max_word_len  # type: int

        # Параметры для архитектуры сети.
        self.input_size = input_size  # type: int
        self.lstm_units = lstm_units  # type: int
        self.embeddings_dimension = embeddings_dimension  # type: int
        self.dense_units = dense_units  # type: int
        self.char_embeddings_dimension = char_embeddings_dimension  # type: int
        self.char_lstm_output_dim = char_lstm_output_dim  # type: int
        self.dropout = dropout  # type: float

        # Словари.
        self.morph = pymorphy2.MorphAnalyzer()  # type: pymorphy2.MorphAnalyzer
        self.grammeme_vectorizer_input = None  # type: GrammemeVectorizer
        self.grammeme_vectorizer_output = None  # type: GrammemeVectorizer

        self.model = None  # type: Model

    def prepare(self, gram_dump_path_input: str, gram_dump_path_output: str, filenames: List[str]=None) -> None:
        """
        Подготовка векторизатора грамматических значений и словаря слов по корпусу.
        """
        self.grammeme_vectorizer_input = GrammemeVectorizer(gram_dump_path_input)
        self.grammeme_vectorizer_output = GrammemeVectorizer(gram_dump_path_output)
        if self.grammeme_vectorizer_input.is_empty() or self.grammeme_vectorizer_output.is_empty():
            loader = Loader(gram_dump_path_input, gram_dump_path_output)
            self.grammeme_vectorizer_input, self.grammeme_vectorizer_output = loader.parse_corpora(filenames)
            self.grammeme_vectorizer_input.save()
            self.grammeme_vectorizer_output.save()

    def save(self, model_config_path: str, model_weights_path: str):
        with open(model_config_path, "w") as f:
            f.write(self.model.to_yaml())
        self.model.save_weights(model_weights_path)

    def load(self, model_config_path: str, model_weights_path: str) -> None:
        with open(model_config_path, "r") as f:
            self.model = model_from_yaml(f.read())
        self.model.load_weights(model_weights_path)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    def build(self):
        """
        Описание модели.
        """
        # Вход граммем
        grammemes = Input(shape=(None, self.grammeme_vectorizer_input.grammemes_count()), name='grammemes')

        # Вход символов
        chars = Input(shape=(None, self.max_word_len), name='chars')
        chars_embedding = Embedding(len(CHAR_SET) + 1, self.char_embeddings_dimension, name='char_embeddings')(chars)
        chars_lstm = TimeDistributed(Bidirectional(
            LSTM(self.char_lstm_output_dim // 2, dropout=self.dropout, recurrent_dropout=self.dropout,
                 return_sequences=False, name='CharLSTM')))(chars_embedding)

        layer = concatenate([grammemes, chars_lstm], name="LSTM_input")
        layer = Bidirectional(LSTM(self.lstm_units, dropout=self.dropout, recurrent_dropout=self.dropout,
                                   return_sequences=True, name='LSTM_1'))(layer)
        layer = Bidirectional(LSTM(self.lstm_units, dropout=self.dropout, recurrent_dropout=self.dropout,
                                   return_sequences=True, name='LSTM_2'))(layer)

        layer = TimeDistributed(Dense(self.dense_units))(layer)
        layer = TimeDistributed(Dropout(self.dropout))(layer)
        layer = TimeDistributed(BatchNormalization())(layer)
        layer = TimeDistributed(Activation('relu'))(layer)

        output = TimeDistributed(Dense(self.grammeme_vectorizer_output.size() + 1, activation='softmax'))(layer)

        self.model = Model(inputs=[grammemes, chars], outputs=[output])

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        print(self.model.summary())

    def train(self, filenames: List[str], model_config_path: str, model_weights_path: str, dump_model_freq: int=1,
              val_part: float=0.33, random_seed: int=42, epochs_num: int=20) -> None:
        """
        Обучение модели.
        
        :param filenames: файлы с морфоразметкой.
        :param model_weights_path: путь, куда сохранять веса модели.
        :param model_config_path: путь, куда сохранять архитектуру модели.
        :param dump_model_freq: насколько часто сохранять модель (1 = каждый батч).
        :param val_part: на какой части выборки оценивать качество.
        :param random_seed: зерно для случайного генератора.
        """
        np.random.seed(random_seed)
        sample_counter = self.count_samples(filenames)
        train_idx, val_idx = self.get_split(sample_counter, val_part)
        for big_epoch in range(epochs_num):
            print('------------Big Epoch {}------------'.format(big_epoch))
            for sentence_len_low, sentence_len_high in self.sentence_len_groups:
                batch_generator = self.get_batch_generator(filenames, sentence_len_low, sentence_len_high, train_idx)
                for epoch, (grammemes, chars, y) in enumerate(batch_generator):
                    self.model.fit([grammemes, chars], y, batch_size=self.nn_batch_size, epochs=1, verbose=2)
                    if epoch != 0 and epoch % dump_model_freq == 0:
                        self.save(model_config_path, model_weights_path)

            self.evaluate(filenames, val_idx)

    @staticmethod
    def count_samples(filenames: List[str]):
        """
        Считает количество предложений в выборке.
        
        :param filenames: файлы выборки.
        :return: количество предложений.
        """
        sample_counter = 0
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        sample_counter += 1
        return sample_counter

    @staticmethod
    def get_split(sample_counter: int, val_part: float) -> Tuple[np.array, np.array]:
        """
        Выдаёт индексы предложений, которые становятся train или val выборкой.
        
        :param sample_counter: количество предложений.
        :param val_part: часть выборки, которая станет val.
        :return: индексы выборок.
        """
        perm = np.random.permutation(sample_counter)
        border = int(sample_counter * (1 - val_part))
        train_idx = perm[:border]
        val_idx = perm[border:]
        return train_idx, val_idx

    def get_batch_generator(self, filenames: List[str], sentence_len_low: int, sentence_len_high: int,
                            indices: np.array) -> BatchGenerator:
        """
        Получение генератора батчей с заданными параметрами.
        
        :param filenames: файлы выборки.
        :param sentence_len_low: нижняя граница длины предложения.
        :param sentence_len_high: верхняя граница длины предложения.
        :param indices: индексы, которые мы разрешаем брать.
        :return: генератор батчей.
        """
        return BatchGenerator(filenames,
                              batch_size=self.external_batch_size,
                              grammeme_vectorizer_input=self.grammeme_vectorizer_input,
                              grammeme_vectorizer_output=self.grammeme_vectorizer_output,
                              sentence_len_low=sentence_len_low,
                              sentence_len_high=sentence_len_high,
                              max_word_len=self.max_word_len,
                              indices=indices)

    def evaluate(self, filenames, val_idx) -> None:
        """
        Оценка на val выборке.
        
        :param filenames: файлы выборки.
        :param val_idx: val индексы.
        """
        word_count = 0
        word_errors = 0
        sentence_count = 0
        sentence_errors = 0
        for sentence_len_low, sentence_len_high in self.sentence_len_groups:
            batch_generator = self.get_batch_generator(filenames, sentence_len_low, sentence_len_high, val_idx)
            for epoch, (grammemes, chars, y) in enumerate(batch_generator):
                predicted_y = self.model.predict([grammemes, chars], batch_size=self.nn_batch_size, verbose=0)
                for i, sentence in enumerate(y):
                    sentence_has_errors = False
                    count_zero = sum([1 for num in sentence if num == [0]])
                    real_sentence_tags = sentence[count_zero:]
                    answer = []
                    for grammeme_probs in predicted_y[i][count_zero:]:
                        num = np.argmax(grammeme_probs)
                        answer.append(num)
                    for tag, predicted_tag in zip(real_sentence_tags, answer):
                        tag = tag[0]
                        word_count += 1
                        if tag != predicted_tag:
                            word_errors += 1
                            sentence_has_errors = True
                    sentence_count += 1
                    if sentence_has_errors:
                        sentence_errors += 1

        print("Word accuracy: ", 1.0 - float(word_errors) / word_count)
        print("Sentence accuracy: ", 1.0 - float(sentence_errors) / sentence_count)

    def predict_proba(self, sentences: List[List[str]]) -> List[List[List[float]]]:
        """
        Предсказание полных PoS-тегов по предложению с вероятностями всех вариантов.
        
        :param sentences: массив предложений (которые являются массивом слов).
        :return: вероятности тегов.
        """
        maxlen = max([len(sentence) for sentence in sentences])
        high_border = 0
        for low, high in self.sentence_len_groups:
            if low <= maxlen <= high:
                high_border = high
        if high_border == 0:
            high_border = maxlen

        n_samples = len(sentences)
        grammemes = np.zeros((n_samples, high_border, self.grammeme_vectorizer_input.grammemes_count()), dtype=np.float)
        chars = np.zeros((n_samples, high_border, self.max_word_len), dtype=np.int)

        for i, sentence in enumerate(sentences):
            gram_vectors, char_vectors = BatchGenerator.get_sample(sentence, self.morph,
                                                                   self.grammeme_vectorizer_input, self.max_word_len)
            grammemes[i, -len(sentence):] = gram_vectors
            chars[i, -len(sentence):] = char_vectors

        return self.model.predict([grammemes, chars])

    def predict(self, sentences: List[List[str]]) -> List[List[int]]:
        """
        Предсказание полных PoS-тегов по предложению.
        
        :param sentences: массив предложений (которые являются массивом слов).
        :return: массив тегов.
        """
        answers = []
        for sentence, probs in zip(sentences, self.predict_proba(sentences)):
            answer = []
            for grammeme_probs in probs[-len(sentence):]:
                num = np.argmax(grammeme_probs[1:])
                answer.append(num)
            answers.append(answer)
        return answers
