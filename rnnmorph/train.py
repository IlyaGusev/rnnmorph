# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Обучение модели с определёнными параметрами.

import os
from typing import List

from rnnmorph.model import LSTMMorphoAnalysis
from rnnmorph.config import BuildModelConfig, TrainConfig
from rnnmorph.util.embeddings import load_embeddings, shrink_w2v
from rnnmorph.settings import RU_TRAIN_CONFIG, RU_BUILD_CONFIG


def train(file_names: List[str], build_config: BuildModelConfig, train_config: TrainConfig):
    model = LSTMMorphoAnalysis()
    model.prepare(train_config.gramm_dict_input, train_config.gramm_dict_output,
                  train_config.word_vocabulary, train_config.char_set_path, file_names)
    if os.path.exists(train_config.model_config_path) and not train_config.rewrite_model:
        model.load(build_config, train_config.model_config_path, train_config.model_weights_path,
                   train_config.train_model_config_path, train_config.train_model_weights_path)
        print(model.eval_model.summary())
    else:
        embeddings = load_embeddings(
            "/media/yallen/My Passport/Models/Vectors/RDT/russian-small-w2v.txt",
            model.word_vocabulary, build_config.word_max_count)
        model.build(build_config, embeddings)
    model.train(file_names, train_config, build_config)


def main():
    # shrink_w2v("/media/yallen/My Passport/Models/Vectors/RDT/russian-big-w2v.txt", 600000,
    #            "/media/yallen/My Passport/Models/Vectors/RDT/russian-small-w2v.txt")
    import sys
    import logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    dir_name = "/media/yallen/My Passport/Datasets/Morpho/clean"
    train_config = TrainConfig()
    train_config.load(RU_TRAIN_CONFIG)
    build_config = BuildModelConfig()
    build_config.load(RU_BUILD_CONFIG)
    file_names = [os.path.join(dir_name, file_name) for file_name in os.listdir(dir_name)]
    train(file_names, build_config, train_config)


if __name__ == "__main__":
    main()