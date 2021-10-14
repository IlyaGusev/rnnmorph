# rnnmorph
[![Current version on PyPI](http://img.shields.io/pypi/v/rnnmorph.svg)](https://pypi.python.org/pypi/rnnmorph)
[![Python versions](https://img.shields.io/pypi/pyversions/rnnmorph.svg)](https://pypi.python.org/pypi/rnnmorph)
[![Tests Status](https://github.com/IlyaGusev/rnnmorph/actions/workflows/python-package.yml/badge.svg)](https://github.com/IlyaGusev/rnnmorph/actions/workflows/python-package.yml)
[![Code Climate](https://codeclimate.com/github/IlyaGusev/rnnmorph/badges/gpa.svg)](https://codeclimate.com/github/IlyaGusev/rnnmorph)

Morphological analyzer (POS tagger) for Russian and English languages based on neural networks and dictionary-lookup systems (pymorphy2, nltk).

### Russian language, MorphoRuEval-2017 test dataset, accuracy

| Domain       | Full tag | PoS tag | F.t. + lemma | Sentence f.t.| Sentence f.t.l. |
|:-------------|:---------|:--------|:-------------|:-------------|:----------------|
| Lenta (news) | 96.31%   | 98.01%  | 92.96%       | 77.93%       | 52.79%          |
| VK (social)  | 95.20%   | 98.04%  | 92.06%       | 74.30%       | 60.56%          |
| JZ (lit.)    | 95.87%   | 98.71%  | 90.45%       | 73.10%       | 43.15%          |
| **All**      | **95.81%**| **98.26%**  | N/A     | **74.92%**   | N/A             |

### English language, UD EWT test, accuracy
| Dataset      | Full tag | PoS tag | F.t. + lemma | Sentence f.t.| Sentence f.t.l. |
|:-------------|:---------|:--------|:-------------|:-------------|:----------------|
| UD EWT test  | 91.57%   | 94.10%  | 87.02%       | 63.17%       | 50.99%          |

### Speed and memory consumption
Speed: from 200 to 600 words per second using CPU. 

Memory consumption: about 500-600 MB for single-sentence predictions

### Install ###
```
sudo pip3 install rnnmorph
```
  
### Usage ###
```
from rnnmorph.predictor import RNNMorphPredictor
predictor = RNNMorphPredictor(language="ru")
forms = predictor.predict(["мама", "мыла", "раму"])
print(forms[0].pos)
>>> NOUN
print(forms[0].tag)
>>> Case=Nom|Gender=Fem|Number=Sing
print(forms[0].normal_form)
>>> мама
print(forms[0].vector)
>>> [0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 1 0 0 1]
```

### Training ###
Simple model training:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Rh46pHS3FP8NHqdux1o2AF32U5xn1V5J)

### Acknowledgements ###
* Anastasyev D. G., Gusev I. O., Indenbom E. M., 2018, [Improving Part-of-speech Tagging Via Multi-task Learning and Character-level Word Representations](http://www.dialog-21.ru/media/4282/anastasyevdg.pdf)
* Anastasyev D. G., Andrianov A. I., Indenbom E. M., 2017, [Part-of-speech Tagging with Rich Language Description](http://www.dialog-21.ru/media/3895/anastasyevdgetal.pdf), [презентация](http://www.dialog-21.ru/media/4102/anastasyev.pdf)
* [Дорожка по морфологическому анализу "Диалога-2017"](http://www.dialog-21.ru/evaluation/2017/morphology/)
* [Материалы дорожки](https://github.com/dialogue-evaluation/morphoRuEval-2017)
* [Morphine by kmike](https://github.com/kmike/morphine), [CRF classifier for MorphoRuEval-2017 by kmike](https://github.com/kmike/dialog2017)
* [Universal Dependencies](http://universaldependencies.org/)
* Tobias Horsmann and Torsten Zesch, 2017, [Do LSTMs really work so well for PoS tagging? – A replication study](http://www.ltl.uni-due.de/wp-content/uploads/horsmannZesch_emnlp2017.pdf)
* Barbara Plank, Anders Søgaard, Yoav Goldberg, 2016, [Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss](https://arxiv.org/abs/1604.05529)
