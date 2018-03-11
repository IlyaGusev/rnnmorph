# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Обучение модели с определёнными параметрами.

import os
from typing import List, Tuple

from rnnmorph.model import LSTMMorphoAnalysis


def train(filenames: List[str], model_config_path: str, model_weights_path: str, gramm_dict_input: str,
          gramm_dict_output: str, rewrite_model: bool=False, input_size: int=5000, external_batch_size: int=2000,
          num_words_in_batch: int=2000, sentence_len_groups: Tuple = ((1, 6), (7, 14), (15, 25), (26, 40), (40, 50)),
          lstm_units=128, embeddings_dimension: int=150, dense_units: int=128, val_part: float=0.1,
          max_word_len: int=30, char_embeddings_dimension: int=10, char_lstm_output_dim: int=64,
          epochs_num: int=20, dropout: float=0.3):
    lstm = LSTMMorphoAnalysis()
    lstm.prepare(gramm_dict_input, gramm_dict_output, filenames)
    if os.path.exists(model_config_path) and not rewrite_model:
        lstm.load(model_config_path, model_weights_path)
        print(lstm.model.summary())
    else:
        lstm.build(input_size=input_size,
                   lstm_units=lstm_units,
                   embeddings_dimension=embeddings_dimension,
                   dense_units=dense_units,
                   max_word_len=max_word_len,
                   char_embeddings_dimension=char_embeddings_dimension,
                   char_lstm_output_dim=char_lstm_output_dim,
                   dropout=dropout)
    lstm.train(filenames, model_config_path, model_weights_path,
               val_part=val_part,
               epochs_num=epochs_num,
               external_batch_size=external_batch_size,
               sentence_len_groups=sentence_len_groups,
               max_word_len=max_word_len,
               num_words_in_batch=num_words_in_batch,
               dump_model_freq=1,
               random_seed=42)


# if __name__ == "__main__":
#     logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#     from rnnmorph.settings import RU_MORPH_DEFAULT_MODEL_CONFIG, RU_MORPH_DEFAULT_MODEL_WEIGHTS, \
#         RU_MORPH_GRAMMEMES_DICT, RU_MORPH_GRAMMEMES_DICT_OUTPUT
#     dir_name = "/media/data/Datasets/Morpho/clean"
#     filenames = [os.path.join(dir_name, filename) for filename in os.listdir(dir_name)]
#     train(filenames,  RU_MORPH_DEFAULT_MODEL_CONFIG, RU_MORPH_DEFAULT_MODEL_WEIGHTS, RU_MORPH_GRAMMEMES_DICT,
#           RU_MORPH_GRAMMEMES_DICT_OUTPUT, val_part=0.1, epochs_num=1)
