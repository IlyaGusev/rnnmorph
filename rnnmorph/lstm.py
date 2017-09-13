from itertools import islice
from typing import List, Tuple

import numpy as np
import pymorphy2
from keras.layers import Input, Embedding, Dense, LSTM, BatchNormalization, Activation, \
    concatenate, Bidirectional, TimeDistributed, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from russian_tagsets import converters

from rnnmorph.data.grammeme_vectorizer import GrammemeVectorizer
from rnnmorph.data.word_form import WordForm
from rnnmorph.loader import WordVocabulary, Loader, process_tag
from rnnmorph.utils.tqdm_open import tqdm_open


class BatchGenerator:
    """
    Генератор наборов примеров для обучения.
    """

    def __init__(self, filenames: List[str], batch_size: int,
                 input_size: int, sentence_maxlen: int, word_vocabulary: WordVocabulary,
                 grammeme_vectorizer_input: GrammemeVectorizer, grammeme_vectorizer_output: GrammemeVectorizer,
                 is_train: bool=True, val_indices: np.array=list()):
        self.filenames = filenames  # type: List[str]
        self.batch_size = batch_size  # type: int
        self.input_size = input_size  # type: int
        self.sentence_maxlen = sentence_maxlen  # type: int
        self.word_vocabulary = word_vocabulary  # type: WordVocabulary
        self.grammeme_vectorizer_input = grammeme_vectorizer_input  # type: GrammemeVectorizer
        self.grammeme_vectorizer_output = grammeme_vectorizer_output  # type: GrammemeVectorizer
        self.val_indices = set(list(val_indices))
        self.is_train = is_train
        self.morph = pymorphy2.MorphAnalyzer()

    def enable_val(self, is_val=True):
        self.is_train = not is_val

    def __to_tensor(self, sentences: List[List[WordForm]]) -> Tuple[np.array, np.array, np.array]:
        n_samples = len(sentences)
        words = np.zeros((n_samples, self.sentence_maxlen), dtype=np.int)
        grammemes = np.zeros((n_samples, self.sentence_maxlen, self.grammeme_vectorizer_input.grammemes_count()), dtype=np.float)
        y = np.zeros((n_samples, self.sentence_maxlen), dtype=np.int)
        i = 0
        for sentence in sentences:
            if len(sentence) <= 1:
                continue
            sentence = sentence[:self.sentence_maxlen]
            texts = [x.text for x in sentence]
            word_indices, gram_vectors = self.get_sample(texts, self.morph, self.grammeme_vectorizer_input,
                                                         self.word_vocabulary, self.input_size)
            assert len(word_indices) == len(sentence)
            assert len(gram_vectors) == len(sentence)
            words[i, -len(sentence):] = word_indices
            grammemes[i, -len(sentence):] = gram_vectors
            y[i, -len(sentence):] = [word.gram_vector_index + 1 for word in sentence]
            i += 1
        y = y.reshape(y.shape[0], y.shape[1], 1)
        return words, grammemes,  y

    @staticmethod
    def get_sample(sentence: List[str], morph, grammeme_vectorizer, word_vocabulary, input_size):
        to_ud = converters.converter('opencorpora-int', 'ud14')
        gram_vectors = []
        for word in sentence:
            gram_value_indices = np.zeros(grammeme_vectorizer.grammemes_count())
            for parse in morph.parse(word):
                pos, gram = process_tag(to_ud, parse.tag, word)
                gram_value_indices += parse.score * np.array(grammeme_vectorizer.get_vector(pos + "#" + gram))
            gram_vectors.append(gram_value_indices)
        word_indices = [min(word_vocabulary.word_to_index[word.lower()]
                            if word in word_vocabulary.word_to_index else input_size,
                            input_size) for word in sentence]
        return word_indices, gram_vectors

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
                        if self.is_train and i in self.val_indices or not self.is_train and i not in self.val_indices:
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
                 sentence_maxlen: int=50, lstm_units=128, embeddings_dimension: int=150, dense_units: int=128):
        self.input_size = input_size  # type: int
        self.external_batch_size = external_batch_size  # type: int
        self.nn_batch_size = nn_batch_size  # type: int
        self.sentence_maxlen = sentence_maxlen  # type: int
        self.word_vocabulary = None  # type: WordVocabulary
        self.grammeme_vectorizer_input = None  # type: GrammemeVectorizer
        self.grammeme_vectorizer_output = None  # type: GrammemeVectorizer
        self.lstm_units = lstm_units  # type: int
        self.embeddings_dimension = embeddings_dimension  # type: int
        self.dense_units = dense_units  # type: int
        self.model = None  # type: Model
        self.morph = pymorphy2.MorphAnalyzer()

    def prepare(self, word_vocab_dump_path: str, gram_dump_path_input: str,
                gram_dump_path_output: str, filenames: List[str]=None) -> None:
        """
        Подготовка векторизатора грамматических значений и словаря слов по корпусу.

        :param word_vocab_dump_path: путь к дампу словаря слов.
        :param filenames: имена файлов с морфоразметкой.
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
        words_embedding = Embedding(self.input_size + 1, self.embeddings_dimension, name='embeddings')(words)

        # Вход граммем
        grammemes_input = Input(shape=(None, self.grammeme_vectorizer_input.grammemes_count()), name='grammemes')

        layer = concatenate([words_embedding, grammemes_input], name="LSTM_input")
        layer = Bidirectional(LSTM(self.lstm_units, dropout=.2, recurrent_dropout=.2,
                                   return_sequences=True, name='LSTM_1'))(layer)
        layer = Bidirectional(LSTM(self.lstm_units, dropout=.2, recurrent_dropout=.2,
                                   return_sequences=True, name='LSTM_2'))(layer)

        layer = TimeDistributed(Dense(self.dense_units))(layer)
        layer = TimeDistributed(Dropout(.2))(layer)
        layer = TimeDistributed(BatchNormalization())(layer)
        layer = TimeDistributed(Activation('relu'))(layer)

        output = TimeDistributed(Dense(self.grammeme_vectorizer_output.size() + 1, activation='softmax'))(layer)

        self.model = Model(inputs=[words, grammemes_input], outputs=[output])

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

    def train(self, filenames: List[str], save_path: str, dump_model_freq: int = 1, val_part: float=0.33) -> None:
        sample_counter = 0
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        sample_counter += 1
        np.random.seed(42)
        perm = np.random.permutation(sample_counter)
        val_idx = perm[int(sample_counter * (1 - val_part)):]

        batch_generator = BatchGenerator(filenames,
                                         batch_size=self.external_batch_size,
                                         grammeme_vectorizer_input=self.grammeme_vectorizer_input,
                                         grammeme_vectorizer_output=self.grammeme_vectorizer_output,
                                         word_vocabulary=self.word_vocabulary,
                                         input_size=self.input_size,
                                         sentence_maxlen=self.sentence_maxlen,
                                         is_train=True,
                                         val_indices=val_idx)

        for big_epoch in range(0, 1000):
            print('------------Big Epoch {}------------'.format(big_epoch))
            batch_generator.enable_val(False)
            for epoch, (words, grammemes, y) in enumerate(batch_generator):
                self.model.fit([words, grammemes], y, batch_size=self.nn_batch_size, epochs=1, verbose=2)
                if epoch != 0 and epoch % dump_model_freq == 0:
                    self.model.save(save_path)

            self.evaluate(batch_generator)

    def evaluate(self, batch_generator):
        word_count = 0
        word_errors = 0
        sentence_count = 0
        sentence_errors = 0
        batch_generator.enable_val(True)
        for epoch, (words, grammemes, y) in enumerate(batch_generator):
            predicted_y = self.model.predict([words, grammemes], batch_size=self.nn_batch_size, verbose=0)
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
        word_indices, gram_vectors = BatchGenerator.get_sample(sentence, self.morph, self.grammeme_vectorizer_input,
                                                               self.word_vocabulary, self.input_size)
        words = np.zeros((1, self.sentence_maxlen), dtype=np.int)
        grammemes = np.zeros((1, self.sentence_maxlen, self.grammeme_vectorizer_input.grammemes_count()), dtype=np.float)
        words[0, -len(sentence):] = word_indices
        grammemes[0, -len(sentence):] = gram_vectors
        answer = []
        for grammeme_probs in self.model.predict([words, grammemes])[0][-len(sentence):]:
            num = np.argmax(grammeme_probs[1:])
            answer.append(self.grammeme_vectorizer_output.get_name_by_index(num))
        return answer
