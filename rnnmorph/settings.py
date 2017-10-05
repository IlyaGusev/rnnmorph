from pkg_resources import resource_filename

RU_MORPH_DEFAULT_MODEL_CONFIG = resource_filename(__name__, "models/model_config.yaml")
RU_MORPH_DEFAULT_MODEL_WEIGHTS = resource_filename(__name__, "models/model_weights.h5")
RU_MORPH_GRAMMEMES_DICT = resource_filename(__name__, "models/gram_input.json")
RU_MORPH_GRAMMEMES_DICT_OUTPUT = resource_filename(__name__, "models/gram_output.json")

TEST_TAGGED_FOLDER = resource_filename(__name__, "test/tagged")
TEST_UNTAGGED_VK = resource_filename(__name__, "test/untagged/VK_extracted.txt")
TEST_UNTAGGED_LENTA = resource_filename(__name__, "test/untagged/Lenta_extracted.txt")
TEST_UNTAGGED_JZ = resource_filename(__name__, "test/untagged/JZ_extracted.txt")
TEST_TAGGED_VK = resource_filename(__name__, "test/tagged/VK_extracted.txt")
TEST_TAGGED_LENTA = resource_filename(__name__, "test/tagged/Lenta_extracted.txt")
TEST_TAGGED_JZ = resource_filename(__name__, "test/tagged/JZ_extracted.txt")
TEST_GOLD_VK = resource_filename(__name__, "test/gold/VK_gold.txt")
TEST_GOLD_LENTA = resource_filename(__name__, "test/gold/Lenta_gold.txt")
TEST_GOLD_JZ = resource_filename(__name__, "test/gold/JZ_gold.txt")