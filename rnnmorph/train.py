import os
from typing import List, Tuple

from rnnmorph.lstm import LSTMMorphoAnalysis


def train(filenames: List[str], model_path: str, word_dict_path: str, gramm_dict_input: str, gramm_dict_output: str,
          rewrite_model: bool=False, input_size: int=5000, external_batch_size: int=20000, nn_batch_size: int=256,
          sentence_len_groups: Tuple = ((1, 6), (7, 14), (15, 25), (26, 40), (40, 50)), lstm_units=128,
          embeddings_dimension: int=150, dense_units: int=128):
    lstm = LSTMMorphoAnalysis(input_size=input_size,
                              external_batch_size=external_batch_size,
                              nn_batch_size=nn_batch_size,
                              sentence_len_groups=sentence_len_groups,
                              lstm_units=lstm_units,
                              embeddings_dimension=embeddings_dimension,
                              dense_units=dense_units)
    lstm.prepare(word_dict_path, gramm_dict_input, gramm_dict_output, filenames)
    if os.path.exists(model_path) and not rewrite_model:
        lstm.load(model_path)
    else:
        lstm.build()
    lstm.train(filenames, model_path)


if __name__ == "__main__":
    from rnnmorph.settings import RU_MORPH_DEFAULT_MODEL, RU_MORPH_WORD_VOCAB_DUMP, \
        RU_MORPH_GRAMMEMES_DICT, RU_MORPH_GRAMMEMES_DICT_OUTPUT
    dir_name = "/media/data/Datasets/Morpho/clean"
    filenames = [os.path.join(dir_name, filename) for filename in os.listdir(dir_name)]
    train(filenames,  RU_MORPH_DEFAULT_MODEL, RU_MORPH_WORD_VOCAB_DUMP,
          RU_MORPH_GRAMMEMES_DICT, RU_MORPH_GRAMMEMES_DICT_OUTPUT)
