from typing import List, Tuple

import pymorphy2
from russian_tagsets import converters

from rnnmorph.data.grammeme_vectorizer import GrammemeVectorizer
from rnnmorph.data.word_vocabulary import WordVocabulary
from rnnmorph.utils.tqdm_open import tqdm_open


def process_tag(to_ud, tag, text):
    ud_tag = to_ud(str(tag), text)
    pos = ud_tag.split()[0]
    gram = ud_tag.split()[1].split("|")
    dropped = ["Animacy", "Aspect", "NumType"]
    gram = [grammem for grammem in gram if sum([drop in grammem for drop in dropped]) == 0]
    return pos, "|".join(sorted(gram))


class Loader(object):
    """
    Класс для построения GrammemeVectorizer и WordFormVocabulary по корпусу
    """
    def __init__(self, gram_dump_path_input, gram_dump_path_output, word_dump_path):
        self.grammeme_vectorizer_input = GrammemeVectorizer(gram_dump_path_input)
        self.grammeme_vectorizer_output = GrammemeVectorizer(gram_dump_path_output)
        self.word_vocabulary = WordVocabulary(word_dump_path)
        self.morph = pymorphy2.MorphAnalyzer()

    def parse_corpora(self, filenames: List[str]) -> Tuple[GrammemeVectorizer, GrammemeVectorizer, WordVocabulary]:
        """
        Построить WordFormVocabulary, GrammemeVectorizer по корпусу

        :param filenames: пути к файлам корпуса.
        """
        for filename in filenames:
            with tqdm_open(filename, encoding="utf-8") as f:
                for line in f:
                    if line == "\n":
                        continue
                    self.__process_line(line)

        self.grammeme_vectorizer_input.init_possible_vectors()
        self.grammeme_vectorizer_output.init_possible_vectors()
        return self.grammeme_vectorizer_input, self.grammeme_vectorizer_output, self.word_vocabulary

    def __process_line(self, line: str) -> None:
        text, lemma, pos_tag, grammemes = line.strip().split("\t")[0:4]
        self.word_vocabulary.add_word(text.lower())
        self.grammeme_vectorizer_output.add_grammemes(pos_tag, grammemes)
        to_ud = converters.converter('opencorpora-int', 'ud14')
        for parse in self.morph.parse(text):
            pos, gram = process_tag(to_ud, parse.tag, text)
            self.grammeme_vectorizer_input.add_grammemes(pos, gram)
