import logging
import sys
import string

import pymorphy2
from russian_tagsets import converters

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from rnnmorph.predictor import MorphPredictor
from rnnmorph.settings import RU_MORPH_DEFAULT_MODEL, RU_MORPH_WORD_VOCAB_DUMP, \
    RU_MORPH_GRAMMEMES_DICT, RU_MORPH_GRAMMEMES_DICT_OUTPUT
from rnnmorph.util.timeit import timeit
from rnnmorph.loader import process_tag


@timeit
def tag(predictor, untagged_filename, tagged_filename):
    morph = pymorphy2.MorphAnalyzer()
    to_ud = converters.converter('opencorpora-int', 'ud14')
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
                    tags = []
                    if len(words) != 0:
                        tags = predictor.predict([word.lower() for word in words])
                    for number, p in enumerate(pos):
                        word = original_words[number]
                        pos_tag = tags[p].split("#")[0]
                        gram = tags[p].split("#")[1]
                        pos_lemma = ""
                        full_lemma = ""
                        first_lemma = morph.parse(word)[0].normal_form
                        for word_form in morph.parse(word):
                            word_form_pos_tag, word_form_gram = process_tag(to_ud, word_form.tag, word)
                            if word_form_pos_tag == pos_tag:
                                pos_lemma = word_form.normal_form
                                if len(set(word_form_gram.split("|")).intersection(set(gram.split("|")))) == \
                                        min(len(gram.split("|")), len(word_form_gram.split("|"))):
                                    full_lemma = word_form.normal_form
                        lemma = first_lemma
                        if full_lemma != "":
                            lemma = full_lemma
                        elif pos_lemma != "":
                            lemma = pos_lemma
                        w.write(str(number+1) + "\t" + word + "\t" + lemma + "\t" + pos_tag + "\t" + gram + "\n")
                    words = []
                    pos = []
                    original_words = []
                    w.write("\n")

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    predictor = MorphPredictor(RU_MORPH_DEFAULT_MODEL, RU_MORPH_WORD_VOCAB_DUMP,
                               RU_MORPH_GRAMMEMES_DICT, RU_MORPH_GRAMMEMES_DICT_OUTPUT)

    tag(predictor, "/home/yallen/Документы/Python/rnnmorph/rnnmorph/test/untagged/Lenta_extracted.txt",
             "/home/yallen/Документы/Python/rnnmorph/rnnmorph/test/tagged/Lenta_extracted.txt")
    tag(predictor, "/home/yallen/Документы/Python/rnnmorph/rnnmorph/test/untagged/VK_extracted.txt",
        "/home/yallen/Документы/Python/rnnmorph/rnnmorph/test/tagged/VK_extracted.txt")
    tag(predictor, "/home/yallen/Документы/Python/rnnmorph/rnnmorph/test/untagged/JZ_extracted.txt",
        "/home/yallen/Документы/Python/rnnmorph/rnnmorph/test/tagged/JZ_extracted.txt")
