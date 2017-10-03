from pkg_resources import resource_filename

RU_MORPH_DEFAULT_MODEL_CONFIG = resource_filename(__name__, "models/model_config.yaml")
RU_MORPH_DEFAULT_MODEL_WEIGHTS = resource_filename(__name__, "models/model_weights.h5")
RU_MORPH_GRAMMEMES_DICT = resource_filename(__name__, "models/gram_input.json")
RU_MORPH_GRAMMEMES_DICT_OUTPUT = resource_filename(__name__, "models/gram_output.json")