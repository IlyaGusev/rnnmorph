from typing import List
from rnnmorph.lstm import LSTMMorphoAnalysis


class MorphPredictor:
    def __init__(self, model_filename: str, word_vocab_filename: str,
                 gramm_dict_input: str, gramm_dict_output: str):
        self.model = LSTMMorphoAnalysis()
        self.model.prepare(word_vocab_filename, gramm_dict_input, gramm_dict_output)
        self.model.load(model_filename)

    def predict(self, words: List[str]):
        return self.model.predict(words)