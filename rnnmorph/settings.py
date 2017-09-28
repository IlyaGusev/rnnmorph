from pkg_resources import resource_filename

RU_MORPH_DEFAULT_MODEL = resource_filename(__name__, "models/morph_model.h5")
RU_MORPH_GRAMMEMES_DICT = resource_filename(__name__, "models/gram_input.json")
RU_MORPH_GRAMMEMES_DICT_OUTPUT = resource_filename(__name__, "models/gram_output.json")
RU_MORPH_WORD_VOCAB_DUMP = resource_filename(__name__, "models/word_vocab.json")
