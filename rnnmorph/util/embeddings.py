# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Вспомогательные функции для работы с словными эмбеддингами.

import numpy as np

from rnnmorph.data_preparation.word_vocabulary import WordVocabulary


def shrink_w2v(input_filename, border, output_filename, print_step=10000):
    vocabulary_embeddings = dict()
    all_count = 0
    correct_count = 0
    with open(input_filename, "r", encoding='utf-8', errors='ignore') as r:
        line = next(r)
        dimension = int(line.strip().split()[1])
        for line in r:
            if all_count % print_step == 0:
                print("Parsed words: {}".format(all_count))
            if correct_count == border:
                break
            all_count += 1
            try:
                word = line.strip().split()[0]
                embedding = [float(i) for i in line.strip().split()[1:]]
                vocabulary_embeddings[word] = embedding
                correct_count += 1
            except ValueError or UnicodeDecodeError:
                continue
        vocabulary_embeddings = {key: value for key, value in vocabulary_embeddings.items() if len(value) == dimension}
    with open(output_filename, "w", encoding='utf-8') as w:
        w.write(str(len(vocabulary_embeddings.items())) + " " + str(dimension) + "\n")
        for word, embedding in vocabulary_embeddings.items():
            embedding = " ".join([str(j) for j in list(embedding)])
            w.write(word + " " + embedding + "\n")


def load_embeddings(embeddings_file_name: str, vocabulary: WordVocabulary, word_count: int):
    with open(embeddings_file_name, "r", encoding='utf-8') as f:
        line = next(f)
        dimension = int(line.strip().split()[1])
        matrix = np.random.rand(min(vocabulary.size(), word_count+1), dimension) * 0.05
        words = {word: i for i, word in enumerate(vocabulary.words[:word_count])}
        for line in f:
            try:
                word = line.strip().split()[0]
                embedding = [float(i) for i in line.strip().split()[1:]]
                index = words.get(word)
                if index is not None:
                    matrix[index] = embedding
            except ValueError or UnicodeDecodeError:
                continue
        return matrix
