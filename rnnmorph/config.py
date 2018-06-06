# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Конфиги для архитектуры модели и процесса обучения.

import json
import copy
from rnnmorph.settings import RU_MODEL_CONFIG, RU_MODEL_WEIGHTS, \
    RU_GRAMMEMES_DICT_INPUT, RU_GRAMMEMES_DICT_OUTPUT, RU_CHAR_MODEL_CONFIG, \
    RU_CHAR_MODEL_WEIGHTS, RU_WORD_VOCABULARY, RU_CHAR_SET, RU_TRAIN_MODEL_CONFIG, \
    RU_TRAIN_MODEL_WEIGHTS


class BuildModelConfig(object):
    def __init__(self):
        self.use_gram = True
        self.gram_hidden_size = 30
        self.gram_dropout = 0.3

        self.use_chars = True
        self.char_max_word_length = 30  # максимальный учитываемый моделью размер слова.
        self.char_embedding_dim = 10  # размерность буквенных эмбеддингов.
        self.char_function_hidden_size = 128
        self.char_dropout = 0.3
        self.char_function_output_size = 64  # размерность эмбеддинга слова, собранного на основе буквенных.

        self.use_word_embeddings = False
        self.word_embedding_dropout = 0.2
        self.word_max_count = 10000
        self.use_trained_char_embeddings = True
        self.char_model_config_path = RU_CHAR_MODEL_CONFIG
        self.char_model_weights_path = RU_CHAR_MODEL_WEIGHTS

        self.rnn_input_size = 200
        self.rnn_hidden_size = 128  # размер состояния у LSTM слоя. (у BiLSTM = rnn_hidden_size * 2).
        self.rnn_n_layers = 2
        self.rnn_dropout = 0.3
        self.rnn_bidirectional = True

        self.dense_size = 128  # размер выхода скрытого слоя.
        self.dense_dropout = 0.3

        self.use_crf = False
        self.use_pos_lm = True
        self.use_word_lm = False

        if self.use_word_lm:
            assert not self.use_word_embeddings

    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            d = copy.deepcopy(self.__dict__)
            f.write(json.dumps(d, sort_keys=True, indent=4) + "\n")

    def load(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            d = json.loads(f.read())
            self.__dict__.update(d)


class TrainConfig(object):
    def __init__(self):
        self.model_config_path = RU_MODEL_CONFIG
        self.model_weights_path = RU_MODEL_WEIGHTS
        self.train_model_config_path = RU_TRAIN_MODEL_CONFIG
        self.train_model_weights_path = RU_TRAIN_MODEL_WEIGHTS
        self.gramm_dict_input = RU_GRAMMEMES_DICT_INPUT
        self.gramm_dict_output = RU_GRAMMEMES_DICT_OUTPUT
        self.word_vocabulary = RU_WORD_VOCABULARY
        self.char_set_path = RU_CHAR_SET
        self.rewrite_model = True
        self.external_batch_size = 10000  # размер батча, который читается из файлов.
        self.num_words_in_batch = 2000  # количество слов в минибатче.
        self.sentence_len_groups = ((1, 6), (7, 14), (15, 25), (26, 40), (40, 50))  # разбиение на бакеты
        self.val_part = 0.05  # на какой части выборки оценивать качество.
        self.epochs_num = 50  # количество эпох.
        self.dump_model_freq = 1  # насколько часто сохранять модель (1 = каждый батч).
        self.random_seed = 42  # зерно для случайного генератора.

    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            d = copy.deepcopy(self.__dict__)
            f.write(json.dumps(d, sort_keys=True, indent=4) + "\n")

    def load(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            d = json.loads(f.read())
            self.__dict__.update(d)
