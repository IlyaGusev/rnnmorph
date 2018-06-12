# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Обучение модели с определёнными параметрами.

import os
from typing import List

from rnnmorph.model import LSTMMorphoAnalysis
from rnnmorph.config import BuildModelConfig, TrainConfig
from rnnmorph.util.embeddings import load_embeddings
from rnnmorph.settings import MODELS_PATHS


def train(file_names: List[str], train_config_path: str, build_config_path: str,
          language: str, embeddings_path: str=None):
    train_config = TrainConfig()
    train_config.load(train_config_path)
    if train_config.train_model_config_path is None:
        train_config.train_model_config_path = MODELS_PATHS[language]["train_model_config"]
    if train_config.train_model_weights_path is None:
        train_config.train_model_weights_path = MODELS_PATHS[language]["train_model_weights"]
    if train_config.eval_model_config_path is None:
        train_config.eval_model_config_path = MODELS_PATHS[language]["eval_model_config"]
    if train_config.eval_model_weights_path is None:
        train_config.eval_model_weights_path = MODELS_PATHS[language]["eval_model_weights"]
    if train_config.gram_dict_input is None:
        train_config.gram_dict_input = MODELS_PATHS[language]["gram_input"]
    if train_config.gram_dict_output is None:
        train_config.gram_dict_output = MODELS_PATHS[language]["gram_output"]
    if train_config.word_vocabulary is None:
        train_config.word_vocabulary = MODELS_PATHS[language]["word_vocabulary"]
    if train_config.char_set_path is None:
        train_config.char_set_path = MODELS_PATHS[language]["char_set"]

    build_config = BuildModelConfig()
    build_config.load(build_config_path)
    if build_config.char_model_weights_path is None:
        build_config.char_model_weights_path = MODELS_PATHS[language]["char_model_weights"]
    if build_config.char_model_config_path is None:
        build_config.char_model_config_path = MODELS_PATHS[language]["char_model_config"]

    model = LSTMMorphoAnalysis(language)
    model.prepare(train_config.gram_dict_input, train_config.gram_dict_output,
                  train_config.word_vocabulary, train_config.char_set_path, file_names)
    if os.path.exists(train_config.eval_model_config_path) and not train_config.rewrite_model:
        model.load_train(build_config, train_config.train_model_config_path, train_config.train_model_weights_path)
        print(model.eval_model.summary())
    else:
        embeddings = None
        if embeddings_path is not None:
            embeddings = load_embeddings(embeddings_path, model.word_vocabulary, build_config.word_max_count)
        print(embeddings.shape)
        model.build(build_config, embeddings)
    model.train(file_names, train_config, build_config)
