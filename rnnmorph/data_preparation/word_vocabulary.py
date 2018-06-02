# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Словарь.

import pickle
from collections import Counter
from typing import List, Dict


class WordVocabulary:
    def __init__(self):
        self.words = []  # type: List
        self.word_to_index = {}  # type: Dict
        self.counter = Counter()  # type: Counter

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

    def size(self):
        return len(self.words)

    def is_empty(self):
        return not bool(self.words)

    def shrink(self, word_count):
        self.words = self.words[:word_count]
        self.word_to_index = dict()
        for i, word in enumerate(self.words):
            self.word_to_index[word] = i
        self.counter = Counter({word: count for word, count in self.counter.items() if word in self.word_to_index})

    def save(self, dump_filename: str) -> None:
        with open(dump_filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, dump_filename: str) -> None:
        with open(dump_filename, "rb") as f:
            vocab = pickle.load(f)
            self.__dict__.update(vocab.__dict__)