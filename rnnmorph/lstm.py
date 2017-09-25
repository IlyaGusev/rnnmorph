from itertools import islice
from typing import List, Tuple

import numpy as np
import pymorphy2
from keras.layers import Input, Embedding, Dense, LSTM, BatchNormalization, Activation, \
    concatenate, Bidirectional, TimeDistributed, Dropout, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam
from russian_tagsets import converters
from keras import backend as K

from rnnmorph.data.grammeme_vectorizer import GrammemeVectorizer
from rnnmorph.data.word_form import WordForm
from rnnmorph.loader import WordVocabulary, Loader, process_tag
from rnnmorph.util.tqdm_open import tqdm_open

CHAR_SET = " абвгдеёжзийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ" \
                   "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-'\""


class BatchGenerator:
    """
    Генератор наборов примеров для обучения.
    """

    def __init__(self, filenames: List[str], batch_size: int, input_size: int, word_vocabulary: WordVocabulary,
                 grammeme_vectorizer_input: GrammemeVectorizer, grammeme_vectorizer_output: GrammemeVectorizer,
                 sentence_len_low: int, sentence_len_high: int, is_train: bool=True, val_indices: np.array=list()):
        self.filenames = filenames  # type: List[str]
        self.batch_size = batch_size  # type: int
        self.input_size = input_size  # type: int
        self.sentence_len_low = sentence_len_low  # type: int
        self.sentence_len_high = sentence_len_high  # type: int
        self.word_vocabulary = word_vocabulary  # type: WordVocabulary
        self.grammeme_vectorizer_input = grammeme_vectorizer_input  # type: GrammemeVectorizer
        self.grammeme_vectorizer_output = grammeme_vectorizer_output  # type: GrammemeVectorizer
        self.val_indices = set(list(val_indices))
        self.is_train = is_train
        self.max_word_len = 40
        self.morph = pymorphy2.MorphAnalyzer()

    def __to_tensor(self, sentences: List[List[WordForm]]) -> Tuple[np.array, np.array, np.array, np.array]:
        n_samples = len(sentences)
        words = np.zeros((n_samples, self.sentence_len_high), dtype=np.int)
        grammemes = np.zeros((n_samples, self.sentence_len_high,
                              self.grammeme_vectorizer_input.grammemes_count()), dtype=np.float)
        chars = np.zeros((n_samples, self.sentence_len_high, self.max_word_len), dtype=np.int)
        y = np.zeros((n_samples, self.sentence_len_high), dtype=np.int)
        i = 0
        for sentence in sentences:
            if len(sentence) <= 1:
                continue
            texts = [x.text for x in sentence]
            word_indices, gram_vectors, char_vectors = \
                self.get_sample(texts, self.morph, self.grammeme_vectorizer_input,
                                self.word_vocabulary, self.input_size, self.max_word_len)
            assert len(word_indices) == len(sentence)
            assert len(gram_vectors) == len(sentence)
            assert len(char_vectors) == len(sentence)

            words[i, -len(sentence):] = word_indices
            grammemes[i, -len(sentence):] = gram_vectors
            chars[i, -len(sentence):] = char_vectors
            y[i, -len(sentence):] = [word.gram_vector_index + 1 for word in sentence]
            i += 1
        y = y.reshape(y.shape[0], y.shape[1], 1)
        return words, grammemes, chars,  y

    @staticmethod
    def get_sample(sentence: List[str], morph, grammeme_vectorizer, word_vocabulary, input_size, max_word_len):
        to_ud = converters.converter('opencorpora-int', 'ud14')
        gram_vectors = []
        char_vectors = []
        for word in sentence:
            char_indices = np.zeros(max_word_len)
            char_indices[-min(len(word), max_word_len):] = \
                [CHAR_SET.index(ch) if ch in CHAR_SET else len(CHAR_SET) for ch in word][:max_word_len]
            char_vectors.append(char_indices)

            gram_value_indices = np.zeros(grammeme_vectorizer.grammemes_count())
            for parse in morph.parse(word):
                pos, gram = process_tag(to_ud, parse.tag, word)
                gram_value_indices += np.array(grammeme_vectorizer.get_vector(pos + "#" + gram))
            sorted_grammemes = sorted(grammeme_vectorizer.all_grammemes.items(), key=lambda x: x[0])
            index = 0
            for category, values in sorted_grammemes:
                mask = gram_value_indices[index:index+len(values)]
                s = sum(mask)
                gram_value_indices[index:index+len(values)] = mask/s
                index += len(values)
            gram_vectors.append(gram_value_indices)
        word_indices = [min(word_vocabulary.word_to_index[word.lower()]
                            if word in word_vocabulary.word_to_index else input_size,
                            input_size) for word in sentence]
        return word_indices, gram_vectors, char_vectors

    def __iter__(self):
        """
        Получение очередного батча.

        :return: индексы словоформ, грамматические векторы, ответы-индексы.
        """
        sentences = [[]]
        i = 0
        for filename in self.filenames:
            with tqdm_open(filename, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        last_sentence = sentences[-1]
                        is_wrong_sentence = (self.is_train and i in self.val_indices) or \
                                            (not self.is_train and i not in self.val_indices) or \
                                            (len(last_sentence) < self.sentence_len_low) or \
                                            (len(last_sentence) > self.sentence_len_high)
                        if is_wrong_sentence:
                            sentences.pop()
                        if len(sentences) >= self.batch_size:
                            yield self.__to_tensor(sentences)
                            sentences = []
                        sentences.append([])
                        i += 1
                    else:
                        word, lemma, pos, tags = line.split('\t')[0:4]
                        word, lemma = word.lower(), lemma.lower() + '_' + pos
                        tags = "|".join(sorted(tags.strip().split("|")))
                        gram_vector_index = self.grammeme_vectorizer_output.get_index_by_name(pos + "#" + tags)
                        sentences[-1].append(WordForm(lemma, gram_vector_index, word))
        yield self.__to_tensor(sentences)


class LSTMMorphoAnalysis:
    def __init__(self, input_size: int=5000, external_batch_size: int=10000, nn_batch_size: int=256,
                 sentence_len_groups: Tuple=((1, 14), (15, 25), (26, 40), (40, 50)),
                 lstm_units=128, embeddings_dimension: int=150, dense_units: int=128):
        self.input_size = input_size  # type: int
        self.external_batch_size = external_batch_size  # type: int
        self.nn_batch_size = nn_batch_size  # type: int
        self.word_vocabulary = None  # type: WordVocabulary
        self.sentence_len_groups = sentence_len_groups
        self.grammeme_vectorizer_input = None  # type: GrammemeVectorizer
        self.grammeme_vectorizer_output = None  # type: GrammemeVectorizer
        self.lstm_units = lstm_units  # type: int
        self.embeddings_dimension = embeddings_dimension  # type: int
        self.dense_units = dense_units  # type: int
        self.model = None  # type: Model
        self.max_word_len = 40
        self.char_embeddings_dimension = 5
        self.morph = pymorphy2.MorphAnalyzer()

    def prepare(self, word_vocab_dump_path: str, gram_dump_path_input: str,
                gram_dump_path_output: str, filenames: List[str]=None) -> None:
        """
        Подготовка векторизатора грамматических значений и словаря слов по корпусу.
        """
        self.grammeme_vectorizer_input = GrammemeVectorizer(gram_dump_path_input)
        self.grammeme_vectorizer_output = GrammemeVectorizer(gram_dump_path_output)
        self.word_vocabulary = WordVocabulary(word_vocab_dump_path)
        if self.grammeme_vectorizer_input.is_empty() or \
                self.grammeme_vectorizer_output.is_empty() or \
                self.word_vocabulary.is_empty():
            loader = Loader(gram_dump_path_input, gram_dump_path_output, word_vocab_dump_path)
            self.grammeme_vectorizer_input, self.grammeme_vectorizer_output, self.word_vocabulary = \
                loader.parse_corpora(filenames)
            self.grammeme_vectorizer_input.save()
            self.grammeme_vectorizer_output.save()
            self.word_vocabulary.save()

    def load(self, model_filename: str) -> None:
        """
        Загрузка модели.

        :param model_filename: файл с моделью.
        """
        self.model = load_model(model_filename)

    def build(self):
        """
        Описание модели.
        """
        # Вход лемм
        words = Input(shape=(None,), name='words')
        words_embedding = Embedding(self.input_size + 1, self.embeddings_dimension, name='word_embeddings')(words)

        # Вход граммем
        grammemes = Input(shape=(None, self.grammeme_vectorizer_input.grammemes_count()), name='grammemes')

        # Вход символов
        def concat_embeddings(x):
            x = K.concatenate(tuple([x[:, :, i] for i in range(x.shape[2])]))
            return x
        chars = Input(shape=(None, self.max_word_len), name='chars')
        chars_embedding = Embedding(len(CHAR_SET) + 1, self.char_embeddings_dimension, name='char_embeddings')(chars)
        chars_embedding = Lambda(concat_embeddings,
                                 output_shape=(None, self.max_word_len*self.char_embeddings_dimension))(chars_embedding)

        layer = concatenate([words_embedding, grammemes, chars_embedding], name="LSTM_input")
        layer = Bidirectional(LSTM(self.lstm_units, dropout=.2, recurrent_dropout=.2,
                                   return_sequences=True, name='LSTM_1'))(layer)
        layer = Bidirectional(LSTM(self.lstm_units, dropout=.2, recurrent_dropout=.2,
                                   return_sequences=True, name='LSTM_2'))(layer)

        layer = TimeDistributed(Dense(self.dense_units))(layer)
        layer = TimeDistributed(Dropout(.2))(layer)
        layer = TimeDistributed(BatchNormalization())(layer)
        layer = TimeDistributed(Activation('relu'))(layer)

        output = TimeDistributed(Dense(self.grammeme_vectorizer_output.size() + 1, activation='softmax'))(layer)

        self.model = Model(inputs=[words, grammemes, chars], outputs=[output])

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        print(self.model.summary())

    @staticmethod
    def __get_validation_data(batch_generator, size):
        """
        Берет первые size батчей из batch_generator для валидационной выборки
        """
        words_list, grammemes_list, y_list = [], [], []
        for words, grammemes, y in islice(batch_generator, size):
            words_list.append(words)
            grammemes_list.append(grammemes)
            y_list.append(y)
        return np.vstack(words_list), np.vstack(grammemes_list), np.vstack(y_list)

    def train(self, filenames: List[str], save_path: str, dump_model_freq: int = 1,
              val_part: float=0.33, random_seed: int=42) -> None:
        np.random.seed(random_seed)
        sample_counter = self.count_samples(filenames)
        val_idx = self.get_val_indices(sample_counter, val_part)
        for big_epoch in range(0, 1000):
            print('------------Big Epoch {}------------'.format(big_epoch))
            for sentence_len_low, sentence_len_high in self.sentence_len_groups:
                batch_generator = self.get_batch_generator(filenames, sentence_len_low,
                                                           sentence_len_high, is_train=True, val_idx=val_idx)
                for epoch, (words, grammemes, chars, y) in enumerate(batch_generator):
                    self.model.fit([words, grammemes, chars], y, batch_size=self.nn_batch_size, epochs=1, verbose=2)
                    if epoch != 0 and epoch % dump_model_freq == 0:
                        self.model.save(save_path)

            self.evaluate(filenames, val_idx)

    @staticmethod
    def count_samples(filenames: List[str]):
        sample_counter = 0
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        sample_counter += 1
        return sample_counter

    @staticmethod
    def get_val_indices(sample_counter: int, val_part: float):
        perm = np.random.permutation(sample_counter)
        val_idx = perm[int(sample_counter * (1 - val_part)):]
        return val_idx

    def get_batch_generator(self, filenames: List[str], sentence_len_low: int, sentence_len_high: int,
                            is_train: bool, val_idx: np.array):
        return BatchGenerator(filenames,
                              batch_size=self.external_batch_size,
                              grammeme_vectorizer_input=self.grammeme_vectorizer_input,
                              grammeme_vectorizer_output=self.grammeme_vectorizer_output,
                              word_vocabulary=self.word_vocabulary,
                              input_size=self.input_size,
                              sentence_len_low=sentence_len_low,
                              sentence_len_high=sentence_len_high,
                              is_train=is_train,
                              val_indices=val_idx)

    def evaluate(self, filenames, val_idx):
        word_count = 0
        word_errors = 0
        sentence_count = 0
        sentence_errors = 0
        for sentence_len_low, sentence_len_high in self.sentence_len_groups:
            batch_generator = self.get_batch_generator(filenames, sentence_len_low,
                                                       sentence_len_high, is_train=False, val_idx=val_idx)
            for epoch, (words, grammemes, chars, y) in enumerate(batch_generator):
                predicted_y = self.model.predict([words, grammemes, chars], batch_size=self.nn_batch_size, verbose=0)
                for i, sentence in enumerate(y):
                    sentence_has_errors = False
                    count_zero = sum([1 for num in sentence if num == [0]])
                    real_sentence_tags = sentence[count_zero:]
                    answer = []
                    for grammeme_probs in predicted_y[i][count_zero:]:
                        num = np.argmax(grammeme_probs)
                        answer.append(num)
                    for tag, predicted_tag in zip(real_sentence_tags, answer):
                        tag = tag[0]
                        word_count += 1
                        if tag != predicted_tag:
                            word_errors += 1
                            sentence_has_errors = True
                    sentence_count += 1
                    if sentence_has_errors:
                        sentence_errors += 1

        print("Word accuracy: ", 1.0 - float(word_errors) / word_count)
        print("Sentence accuracy: ", 1.0 - float(sentence_errors) / sentence_count)

    def predict(self, sentence: List[str]):
        word_indices, gram_vectors, char_vectors = \
            BatchGenerator.get_sample(sentence, self.morph, self.grammeme_vectorizer_input,
                                      self.word_vocabulary, self.input_size, self.max_word_len)
        high_border = 0
        for low, high in self.sentence_len_groups:
            if low <= len(sentence) <= high:
                high_border = high
        if high_border == 0:
            high_border = len(sentence)
        words = np.zeros((1, high_border), dtype=np.int)
        grammemes = np.zeros((1, high_border, self.grammeme_vectorizer_input.grammemes_count()), dtype=np.float)
        chars = np.zeros((1, high_border, self.max_word_len), dtype=np.int)
        words[0, -len(sentence):] = word_indices
        grammemes[0, -len(sentence):] = gram_vectors
        chars[0, -len(sentence):] = char_vectors
        answer = []
        for grammeme_probs in self.model.predict([words, grammemes, chars])[0][-len(sentence):]:
            num = np.argmax(grammeme_probs[1:])
            answer.append(self.grammeme_vectorizer_output.get_name_by_index(num))
        return answer
