# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Запуск предсказания для жанровых выборок.

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# from rnnmorph.predictor import RNNMorphPredictor
# from rnnmorph.settings import RU_MORPH_DEFAULT_MODEL_CONFIG, RU_MORPH_DEFAULT_MODEL_WEIGHTS, \
#     RU_MORPH_GRAMMEMES_DICT, RU_MORPH_GRAMMEMES_DICT_OUTPUT
from rnnmorph.util.timeit import timeit


@timeit
def tag(predictor, untagged_filename, tagged_filename):
    with open(untagged_filename, "r") as r:
        with open(tagged_filename, "w") as w:
            words = []
            pos = []
            original_words = []
            for line in r:
                if line != "\n":
                    records = line.strip().split("\t")
                    word = records[1]
                    original_words.append(word)
                    pos.append(len(words))
                    words.append(word)
                else:
                    forms = predictor.predict_sentence([word.lower() for word in words])
                    for i, form in enumerate(forms):
                        w.write(str(i+1) + "\t" + form.word + "\t" + form.normal_form + "\t" + form.pos + "\t" + form.tag + "\n")
                    words = []
                    pos = []
                    original_words = []
                    w.write("\n")

# if __name__ == "__main__":
#     logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#     predictor = RNNMorphPredictor(RU_MORPH_DEFAULT_MODEL_CONFIG, RU_MORPH_DEFAULT_MODEL_WEIGHTS,
#                                RU_MORPH_GRAMMEMES_DICT, RU_MORPH_GRAMMEMES_DICT_OUTPUT)
#
#     tag(predictor, "/home/yallen/Документы/Python/rnnmorph/rnnmorph/test/untagged/Lenta_extracted.txt",
#              "/home/yallen/Документы/Python/rnnmorph/rnnmorph/test/tagged/Lenta_extracted.txt")
#     tag(predictor, "/home/yallen/Документы/Python/rnnmorph/rnnmorph/test/untagged/VK_extracted.txt",
#         "/home/yallen/Документы/Python/rnnmorph/rnnmorph/test/tagged/VK_extracted.txt")
#     tag(predictor, "/home/yallen/Документы/Python/rnnmorph/rnnmorph/test/untagged/JZ_extracted.txt",
#         "/home/yallen/Документы/Python/rnnmorph/rnnmorph/test/tagged/JZ_extracted.txt")
