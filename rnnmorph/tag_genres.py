# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Запуск предсказания для жанровых выборок.

import os
from typing import Dict
from rnnmorph.predictor import RNNMorphPredictor
from rnnmorph.settings import TEST_TAGGED_JZ, TEST_TAGGED_LENTA, TEST_TAGGED_VK, TEST_UNTAGGED_JZ, \
    TEST_UNTAGGED_LENTA, TEST_UNTAGGED_VK, TEST_GOLD_JZ, TEST_GOLD_LENTA, TEST_GOLD_VK, TEST_TAGGED_FOLDER
from rnnmorph.util.timeit import timeit
from rnnmorph.test.evaluate import measure


@timeit
def tag(predictor: RNNMorphPredictor, untagged_filename: str, tagged_filename: str):
    sentences = []
    with open(untagged_filename, "r", encoding='utf-8') as r:
        words = []
        for line in r:
            if line != "\n":
                records = line.strip().split("\t")
                word = records[1]
                words.append(word)
            else:
                sentences.append([word.lower() for word in words])
                words = []
    with open(tagged_filename, "w",  encoding='utf-8') as w:
        all_forms = predictor.predict_sentences_tags(sentences)
        for forms in all_forms:
            for i, form in enumerate(forms):
                line = "{}\t{}\t{}\t{}\t{}\n".format(str(i + 1), form.word, form.normal_form, form.pos, form.tag)
                w.write(line)
            w.write("\n")


def tag_files(predictor: RNNMorphPredictor) -> Dict:
    if not os.path.exists(TEST_TAGGED_FOLDER):
        os.makedirs(TEST_TAGGED_FOLDER)
    tag(predictor, TEST_UNTAGGED_LENTA, TEST_TAGGED_LENTA)
    tag(predictor, TEST_UNTAGGED_VK, TEST_TAGGED_VK)
    tag(predictor, TEST_UNTAGGED_JZ, TEST_TAGGED_JZ)

    quality = dict()
    quality['Lenta'] = measure(TEST_GOLD_LENTA, TEST_TAGGED_LENTA, True, None)
    quality['VK'] = measure(TEST_GOLD_VK, TEST_TAGGED_VK, True, None)
    quality['JZ'] = measure(TEST_GOLD_JZ, TEST_TAGGED_JZ, True, None)
    return quality
