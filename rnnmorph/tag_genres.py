# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Запуск предсказания для жанровых выборок.

import os
from typing import Dict
from rnnmorph.predictor import RNNMorphPredictor
from rnnmorph.settings import TEST_TAGGED_JZ, TEST_TAGGED_LENTA, TEST_TAGGED_VK, TEST_UNTAGGED_JZ, \
    TEST_UNTAGGED_LENTA, TEST_UNTAGGED_VK, TEST_GOLD_JZ, TEST_GOLD_LENTA, TEST_GOLD_VK, TEST_TAGGED_FOLDER, \
    TEST_GOLD_EN_EWT_UD, TEST_TAGGED_EN_EWT_UD
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
                sentences.append([word for word in words])
                words = []
    with open(tagged_filename, "w",  encoding='utf-8') as w:
        all_forms = predictor.predict_sentences(sentences)
        for forms in all_forms:
            for i, form in enumerate(forms):
                line = "{}\t{}\t{}\t{}\t{}\n".format(str(i + 1), form.word, form.normal_form, form.pos, form.tag)
                w.write(line)
            w.write("\n")


def tag_ru_files(predictor: RNNMorphPredictor) -> Dict:
    if not os.path.exists(TEST_TAGGED_FOLDER):
        os.makedirs(TEST_TAGGED_FOLDER)
    tag(predictor, TEST_UNTAGGED_LENTA, TEST_TAGGED_LENTA)
    tag(predictor, TEST_UNTAGGED_VK, TEST_TAGGED_VK)
    tag(predictor, TEST_UNTAGGED_JZ, TEST_TAGGED_JZ)

    quality = dict()
    print("Lenta:")
    quality['Lenta'] = measure(TEST_GOLD_LENTA, TEST_TAGGED_LENTA, True, None)
    print("VK:")
    quality['VK'] = measure(TEST_GOLD_VK, TEST_TAGGED_VK, True, None)
    print("JZ:")
    quality['JZ'] = measure(TEST_GOLD_JZ, TEST_TAGGED_JZ, True, None)
    print("All:")
    count_correct_tags = quality['Lenta'].correct_tags + quality['VK'].correct_tags + quality['JZ'].correct_tags
    count_correct_pos = quality['Lenta'].correct_pos + quality['VK'].correct_pos + quality['JZ'].correct_pos
    count_tags = quality['Lenta'].total_tags + quality['VK'].total_tags + quality['JZ'].total_tags
    count_correct_sentences = quality['Lenta'].correct_sentences + quality['VK'].correct_sentences + \
                              quality['JZ'].correct_sentences
    count_sentences = quality['Lenta'].total_sentences + quality['VK'].total_sentences + \
                      quality['JZ'].total_sentences
    quality['All'] = dict()
    quality['All']['tag_accuracy'] = float(count_correct_tags) / count_tags
    quality['All']['pos_accuracy'] = float(count_correct_pos) / count_tags
    quality['All']['sentence_accuracy'] = float(count_correct_sentences) / count_sentences
    return quality


def tag_en_files(predictor: RNNMorphPredictor):
    if not os.path.exists(TEST_TAGGED_FOLDER):
        os.makedirs(TEST_TAGGED_FOLDER)
    tag(predictor, TEST_GOLD_EN_EWT_UD, TEST_TAGGED_EN_EWT_UD)
    return measure(TEST_GOLD_EN_EWT_UD, TEST_TAGGED_EN_EWT_UD, True, None)
