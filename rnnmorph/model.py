# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модель PoS-теггера на основе BiLSTM.

from typing import List, Tuple

import numpy as np
import pymorphy2
from keras.layers import Input, Embedding, Dense, LSTM, BatchNormalization, Activation, \
    concatenate, Bidirectional, TimeDistributed, Dropout, Flatten, Reshape
from keras.models import Model, model_from_yaml
from keras.optimizers import Adam

from rnnmorph.batch_generator import BatchGenerator, CHAR_SET
from rnnmorph.data_preparation.grammeme_vectorizer import GrammemeVectorizer
from rnnmorph.data_preparation.loader import Loader


class LSTMMorphoAnalysis:
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()  # type: pymorphy2.MorphAnalyzer
        self.grammeme_vectorizer_input = None  # type: GrammemeVectorizer
        self.grammeme_vectorizer_output = None  # type: GrammemeVectorizer
        self.model = None  # type: Model

    def prepare(self, gram_dump_path_input: str, gram_dump_path_output: str, filenames: List[str] = None) -> None:
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
        with open(model_config_path, "w", encoding='utf-8') as f:
            f.write(self.model.to_yaml())
        self.model.save_weights(model_weights_path)

    def load(self, model_config_path: str, model_weights_path: str) -> None:
        with open(model_config_path, "r", encoding='utf-8') as f:
            self.model = model_from_yaml(f.read())
        self.model.load_weights(model_weights_path)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    def build(self, lstm_units: int, input_size: int, embeddings_dimension: int,
              dense_units: int, char_embeddings_dimension: int, char_lstm_output_dim: int,
              dropout: float, max_word_len: int):
        """
        Описание модели.
        :param lstm_units: размер состояния у LSTM слоя. (у BiLSTM = lstm_units * 2).
        :param input_size: размер набора слова, которые кодируются эмбеддингами.
        :param embeddings_dimension: размерность словных эмбеддингов.
        :param dense_units: размер выхода скрытого слоя.
        :param char_embeddings_dimension: размерность буквенных эмбеддингов.
        :param char_lstm_output_dim: размерность эмбеддинга слова, собранного на основе букевнных.
        :param dropout: дропаут на основных слоях.
        :param max_word_len: максимальный учитываемый моделью размер слова.
        """
        # Вход граммем
        grammemes = Input(shape=(None, self.grammeme_vectorizer_input.grammemes_count()), name='grammemes')

        # Вход символов
        chars = Input(shape=(None, max_word_len), name='chars')
        chars_embedding = Embedding(len(CHAR_SET) + 1, char_embeddings_dimension, name='char_embeddings')(chars)
        chars_embedding = Reshape((-1, char_embeddings_dimension * max_word_len,))(chars_embedding)

        chars_dense = Dense(char_lstm_output_dim, name='char_dense')(chars_embedding)
        chars_dense = Dropout(dropout)(chars_dense)

        layer = concatenate([grammemes, chars_dense], name="LSTM_input")
        layer = Bidirectional(LSTM(lstm_units, dropout=dropout, recurrent_dropout=dropout,
                                   return_sequences=True, name='LSTM_1'))(layer)
        layer = Bidirectional(LSTM(lstm_units, dropout=dropout, recurrent_dropout=dropout,
                                   return_sequences=True, name='LSTM_2'))(layer)

        layer = TimeDistributed(Dense(dense_units))(layer)
        layer = TimeDistributed(Dropout(dropout))(layer)
        layer = TimeDistributed(BatchNormalization())(layer)
        layer = TimeDistributed(Activation('relu'))(layer)

        output = TimeDistributed(Dense(self.grammeme_vectorizer_output.size() + 1, activation='softmax'))(layer)

        self.model = Model(inputs=[grammemes, chars], outputs=[output])

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        print(self.model.summary())

    def train(self, file_names: List[str], model_config_path: str, model_weights_path: str,
              dump_model_freq: int, val_part: float, random_seed: int, epochs_num: int,
              num_words_in_batch: int, external_batch_size: int, sentence_len_groups: Tuple[Tuple[int, int]],
              max_word_len: int) -> None:
        """
        Обучение модели.
        :param file_names: файлы с морфоразметкой.
        :param model_weights_path: путь, куда сохранять веса модели.
        :param model_config_path: путь, куда сохранять архитектуру модели.
        :param dump_model_freq: насколько часто сохранять модель (1 = каждый батч).
        :param val_part: на какой части выборки оценивать качество.
        :param random_seed: зерно для случайного генератора.
        :param external_batch_size: размер батча, который читается из файлов.
        :param epochs_num: количество эпох.
        :param num_words_in_batch: количество слов в минибатче.
        :param sentence_len_groups: разбиение на бакеты
        :param max_word_len: максимальный учитываемый размер слова.
        """
        np.random.seed(random_seed)
        sample_counter = self.count_samples(file_names)
        train_idx, val_idx = self.get_split(sample_counter, val_part)
        for big_epoch in range(epochs_num):
            print('------------Big Epoch {}------------'.format(big_epoch))
            batch_generator = \
                BatchGenerator(file_names,
                               batch_size=external_batch_size,
                               grammeme_vectorizer_input=self.grammeme_vectorizer_input,
                               grammeme_vectorizer_output=self.grammeme_vectorizer_output,
                               bucket_borders=sentence_len_groups,
                               max_word_len=max_word_len,
                               indices=train_idx)
            for epoch, (grammemes, chars, y) in enumerate(batch_generator):
                max_sentence_length = grammemes.shape[1]
                batch_size = num_words_in_batch // int(max_sentence_length)
                self.model.fit([grammemes, chars], y, batch_size=batch_size, epochs=1, verbose=2)
                if epoch != 0 and epoch % dump_model_freq == 0:
                    self.save(model_config_path, model_weights_path)
            self.evaluate(file_names, val_idx,
                          num_words_in_batch=num_words_in_batch,
                          external_batch_size=external_batch_size,
                          bucket_borders=sentence_len_groups,
                          max_word_len=max_word_len)

    @staticmethod
    def count_samples(filenames: List[str]):
        """
        Считает количество предложений в выборке.
        :param filenames: файлы выборки.
        :return: количество предложений.
        """
        sample_counter = 0
        for filename in filenames:
            with open(filename, "r", encoding='utf-8') as f:
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

    def evaluate(self, filenames, val_idx,
                 num_words_in_batch: int,
                 external_batch_size: int,
                 bucket_borders: Tuple[Tuple[int, int]],
                 max_word_len: int) -> None:
        """
        Оценка на val выборке.
        :param filenames: файлы выборки.
        :param val_idx: val индексы.
        :param external_batch_size: размер батча, который читается из файлов.
        :param num_words_in_batch: количество слов в минибатче.
        :param bucket_borders: разбиение на бакеты
        :param max_word_len: максимальный учитываемый размер слова.
        """
        word_count = 0
        word_errors = 0
        sentence_count = 0
        sentence_errors = 0
        batch_generator = \
            BatchGenerator(filenames,
                           batch_size=external_batch_size,
                           grammeme_vectorizer_input=self.grammeme_vectorizer_input,
                           grammeme_vectorizer_output=self.grammeme_vectorizer_output,
                           bucket_borders=bucket_borders,
                           max_word_len=max_word_len,
                           indices=val_idx)
        for epoch, (grammemes, chars, y) in enumerate(batch_generator):
            max_sentence_length = grammemes.shape[1]
            batch_size = num_words_in_batch // max_sentence_length
            predicted_y = self.model.predict([grammemes, chars], batch_size=batch_size, verbose=0)
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

    def predict_proba(self, sentences: List[List[str]], batch_size: int,
                      max_word_len: int=30) -> List[List[List[float]]]:
        """
        Предсказание полных PoS-тегов по предложению с вероятностями всех вариантов.
        :param sentences: массив предложений (которые являются массивом слов).
        :param max_word_len: максимальный учитываемый размер слова.
        :param batch_size: размер батча.
        :return: вероятности тегов.
        """
        max_sentence_len = max([len(sentence) for sentence in sentences])
        n_samples = len(sentences)
        grammemes = np.zeros((n_samples, max_sentence_len, self.grammeme_vectorizer_input.grammemes_count()),
                             dtype=np.float)
        chars = np.zeros((n_samples, max_sentence_len, max_word_len), dtype=np.int)

        for i, sentence in enumerate(sentences):
            gram_vectors, char_vectors = BatchGenerator.get_sample(sentence, self.morph,
                                                                   self.grammeme_vectorizer_input, max_word_len)
            grammemes[i, -len(sentence):] = gram_vectors
            chars[i, -len(sentence):] = char_vectors

        return self.model.predict([grammemes, chars], batch_size=batch_size)

    def predict(self, sentences: List[List[str]], batch_size: int) -> List[List[int]]:
        """
        Предсказание полных PoS-тегов по предложению.
        :param sentences: массив предложений (которые являются массивом слов).
        :param batch_size: размер батча.
        :return: массив тегов.
        """
        answers = []
        for sentence, probs in zip(sentences, self.predict_proba(sentences, batch_size)):
            answer = []
            for grammeme_probs in probs[-len(sentence):]:
                num = np.argmax(grammeme_probs[1:])
                answer.append(num)
            answers.append(answer)
        return answers
