# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Словарь.

import pickle
import os
from collections import Counter
from typing import List, Dict


class WordVocabulary:
    def __init__(self, dump_filename: str):
        self.dump_filename = dump_filename  # type: str
        self.words = []  # type: List
        self.word_to_index = {}  # type: Dict
        self.counter = Counter()  # type: Counter
        if os.path.exists(self.dump_filename):
            self.load()

    def add_word(self, word: str):
        if word in self.word_to_index:
            self.counter[word] += 1
        else:
            self.words.append(word)
            self.counter[word] = 1
            self.word_to_index[word] = len(self.words) - 1

    def has_word(self, word: str) -> bool:
        return word in self.word_to_index

    def sort(self):
        self.words = []
        self.word_to_index = {}
        for word, _ in self.counter.most_common():
            self.words.append(word)
            self.word_to_index[word] = len(self.words) - 1

    def is_empty(self):
        return len(self.words) == 0

    def save(self) -> None:
        with open(self.dump_filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self) -> None:
        with open(self.dump_filename, "rb") as f:
            vocab = pickle.load(f)
            self.__dict__.update(vocab.__dict__)
