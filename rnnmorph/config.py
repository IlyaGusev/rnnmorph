import json
import copy
from rnnmorph.settings import RU_MORPH_DEFAULT_MODEL_CONFIG, RU_MORPH_DEFAULT_MODEL_WEIGHTS, \
    RU_MORPH_GRAMMEMES_DICT_INPUT, RU_MORPH_GRAMMEMES_DICT_OUTPUT, RU_MORPH_DEFAULT_CHAR_MODEL_CONFIG, \
    RU_MORPH_DEFAULT_CHAR_MODEL_WEIGHTS, RU_MORPH_WORD_VOCABULARY


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
        self.word_max_count = 100000
        self.use_trained_char_embeddings = True
        self.char_model_config_path = RU_MORPH_DEFAULT_CHAR_MODEL_CONFIG
        self.char_model_weights_path = RU_MORPH_DEFAULT_CHAR_MODEL_WEIGHTS

        self.rnn_hidden_size = 128  # размер состояния у LSTM слоя. (у BiLSTM = rnn_hidden_size * 2).
        self.rnn_n_layers = 2
        self.rnn_dropout = 0.3
        self.rnn_bidirectional = True

        self.dense_size = 128  # размер выхода скрытого слоя.
        self.dense_dropout = 0.3

        self.use_crf = True

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
        self.model_config_path = RU_MORPH_DEFAULT_MODEL_CONFIG
        self.model_weights_path = RU_MORPH_DEFAULT_MODEL_WEIGHTS
        self.gramm_dict_input = RU_MORPH_GRAMMEMES_DICT_INPUT
        self.gramm_dict_output = RU_MORPH_GRAMMEMES_DICT_OUTPUT
        self.word_vocabulary = RU_MORPH_WORD_VOCABULARY
        self.rewrite_model = True
        self.external_batch_size = 2000  # размер батча, который читается из файлов.
        self.num_words_in_batch = 2000  # количество слов в минибатче.
        self.sentence_len_groups = ((1, 6), (7, 14), (15, 25), (26, 40), (40, 50))  # разбиение на бакеты
        self.val_part = 0.1  # на какой части выборки оценивать качество.
        self.epochs_num = 20  # количество эпох.
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
