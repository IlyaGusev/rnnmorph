from collections import Counter
import pickle
import os


class WordVocabulary:
    def __init__(self, dump_filename):
        self.dump_filename = dump_filename
        self.words = []
        self.word_to_index = {}
        self.counter = Counter()
        if os.path.exists(self.dump_filename):
            self.load()

    def add_word(self, word):
        if word in self.word_to_index:
            self.counter[word] += 1
        else:
            self.words.append(word)
            self.counter[word] = 1
            self.word_to_index[word] = len(self.words) - 1

    def sort(self):
        self.words = []
        self.word_to_index = {}
        for word, _ in self.counter.most_common():
            self.words.append(word)
            self.word_to_index[word] = len(self.words) - 1

    def is_empty(self):
        return len(self.words) == 0

    def save(self) -> None:
        """
        Сохранение словаря.
        """
        with open(self.dump_filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self) -> None:
        """
        Загрузка словаря.
        """
        with open(self.dump_filename, "rb") as f:
            vocab = pickle.load(f)
            self.__dict__.update(vocab.__dict__)