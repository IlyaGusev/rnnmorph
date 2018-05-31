# -*- coding: utf-8 -*-
# Автор: Гусев Илья

from typing import List, Tuple

import numpy as np
from keras.layers import Input, Embedding, Dense, Dropout, Reshape, Activation
from keras.models import Model, model_from_yaml
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from rnnmorph.batch_generator import CHAR_SET


class CharEmbeddingsModel:
    def __init__(self):
        self.model = None  # type: Model

    def build(self,
              char_embeddings_dimension: int,
              vocabulary_size: int,
              word_embeddings_dimension: int,
              max_word_length: int,
              char_dense_1_output_dim: int,
              char_function_output_dim: int,
              dropout: float,
              word_embeddings: np.array):
        chars = Input(shape=(max_word_length, ), name='chars')

        chars_embedding = Embedding(len(CHAR_SET) + 1, char_embeddings_dimension, name='char_embeddings')(chars)
        chars_embedding = Reshape((char_embeddings_dimension * max_word_length,))(chars_embedding)
        chars_dense = Dense(char_dense_1_output_dim, name='char_dense_1')(chars_embedding)
        chars_dense = Dropout(dropout)(chars_dense)
        chars_dense = Dense(char_function_output_dim, name='char_dense_2')(chars_dense)
        chars_dense = Dropout(dropout)(chars_dense)

        predicted_word_embeddings = Dense(word_embeddings_dimension, name='char_embed_to_word_embed')(chars_dense)
        bias = np.zeros((vocabulary_size, ), dtype=np.int)
        output = Dense(vocabulary_size, weights=[word_embeddings, bias],
                       trainable=False, activation='softmax')(predicted_word_embeddings)

        self.model = Model(inputs=[chars], outputs=[output])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())
        print(self.model.summary())

    def train(self,
              vocabulary: List[str],
              val_part: float,
              random_seed: int,
              batch_size: int,
              max_word_len: int) -> None:
        """
        Обучение модели.

        :param vocabulary: список слов.
        :param val_part: на какой части выборки оценивать качество.
        :param random_seed: зерно для случайного генератора.
        :param batch_size: размер батча.
        :param max_word_len: максимальный учитываемый размер слова.
        """
        np.random.seed(random_seed)
        chars, y = self.prepare_words(vocabulary, max_word_len)
        callbacks = [EarlyStopping(patience=1)]
        self.model.fit(chars, y, batch_size=batch_size, epochs=100, verbose=2,
                       validation_split=val_part, callbacks=callbacks)

    @staticmethod
    def prepare_words(words, max_word_length):
        chars = np.zeros((len(words), max_word_length), dtype=np.int)
        y = np.zeros((len(words), ), dtype=np.int)
        for i in range(len(words)):
            y[i] = i
        for i, word in enumerate(words):
            word_char_indices = [CHAR_SET.index(ch) if ch in CHAR_SET else len(CHAR_SET) for ch in word][:max_word_length]
            chars[i, -min(len(word), max_word_length):] = word_char_indices
        return chars, y


def shrink_w2v(input_filename, border, output_filename, print_step=10000):
    vocabulary_embeddings = dict()
    all_count = 0
    correct_count = 0
    with open(input_filename, "r", encoding='utf-8', errors='ignore') as r:
        line = next(r)
        dimension = int(line.strip().split()[1])
        for line in r:
            if all_count % print_step == 0:
                print("Parsed words: {}".format(all_count))
            if correct_count == border:
                break
            all_count += 1
            try:
                word = line.strip().split()[0]
                embedding = [float(i) for i in line.strip().split()[1:]]
                vocabulary_embeddings[word] = embedding
                correct_count += 1
            except ValueError or UnicodeDecodeError:
                continue
        vocabulary_embeddings = {key: value for key, value in vocabulary_embeddings.items() if len(value) == dimension}
    with open(output_filename, "w", encoding='utf-8') as w:
        w.write(str(len(vocabulary_embeddings.items())) + " " + str(dimension) + "\n")
        for word, embedding in vocabulary_embeddings.items():
            embedding = " ".join([str(j) for j in list(embedding)])
            w.write(word + " " + embedding + "\n")


def get_pretrained_char_embeddings(
        embeddings_file_name: str,
        batch_size: int,
        model_weights_path: str,
        val_part: float=0.2,
        seed: int=42):
    """
    :param embeddings_file_name: путь к файлу со словными эмбеддингами.
    :param batch_size: размер батча.
    :param model_weights_path: путь, куда сохранять веса модели.
    :param val_part: доля val выборки.
    :param seed: seed для ГПСЧ
    """
    import gensim
    w2v = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file_name, binary=False)

    vocabulary = list(w2v.vocab.keys())
    np.random.shuffle(vocabulary)

    matrix = np.zeros((len(vocabulary), w2v.vector_size), dtype=np.float)
    for i, word in enumerate(vocabulary):
        matrix[i] = w2v[word]

    max_word_length = 40
    model = CharEmbeddingsModel()
    model.build(char_embeddings_dimension=5,
                vocabulary_size=len(vocabulary),
                word_embeddings_dimension=w2v.vector_size,
                max_word_length=max_word_length,
                char_dense_1_output_dim=100,
                char_function_output_dim=50,
                dropout=0.3,
                word_embeddings=matrix.T)
    model.train(vocabulary, val_part, seed, batch_size, max_word_length)
    model.model.layers.pop()
    model.model.layers.pop()
    model.model.save_weights(model_weights_path)


# shrink_w2v("/media/yallen/My Passport/Models/Vectors/RDT/russian-big-w2v.txt", 10000,
#            "/media/yallen/My Passport/Models/Vectors/RDT/russian-sample-w2v.txt")
get_pretrained_char_embeddings("/media/yallen/My Passport/Models/Vectors/RDT/russian-sample-w2v.txt", 64,
                               "char_model.h5")
