# rnnmorph
[![Current version on PyPI](http://img.shields.io/pypi/v/rnnmorph.svg)](https://pypi.python.org/pypi/rnnmorph)
[![Python versions](https://img.shields.io/pypi/pyversions/rnnmorph.svg)](https://pypi.python.org/pypi/rnnmorph)
[![Build Status](https://travis-ci.org/IlyaGusev/rnnmorph.svg?branch=master)](https://travis-ci.org/IlyaGusev/rnnmorph)
[![Code Climate](https://codeclimate.com/github/IlyaGusev/rnnmorph/badges/gpa.svg)](https://codeclimate.com/github/IlyaGusev/rnnmorph)

Морфологический анализатор на основе нейронных сетей и pymorphy2.

Lenta:
* Качество по тегам:
  * 4020 меток из 4179, точность 96.20%
  * 274 предложений из 358, точность 76.54%
* Качество полного разбора:
  * 3882 слов из 4179, точность 92.89%
  * 186 предложений из 358, точность 51.96%

VK:
* Качество по тегам:
  * 3706 меток из 3877, точность 95.59%
  * 433 предложений из 568, точность 76.23%
* Качество полного разбора:
  * 3583 слов из 3877, точность 92.42%
  * 349 предложений из 568, точность 61.44%

JZ:
* Качество по тегам:
  * 3871 меток из 4042, точность 95.77%
  * 290 предложений из 394, точность 73.60%
* Качество полного разбора:
  * 3652 слов из 4042, точность 90.35%
  * 169 предложений из 394, точность 42.89%

All:
* Точность по тегам по всем разделам: 95.86%
* Точность по предложениям по всем разделам: 75.53%
  
Скорость: от 200 до 600 слов в секунду.

Потребление оперативной памяти: зависит от режима работы, для предсказания одиночных предложений - 500-600 Мб, для режима с батчами - пропорционально размеру батча.

### Install ###
```
sudo pip3 install rnnmorph
```
  
### Usage ###
```
from rnnmorph.predictor import RNNMorphPredictor
predictor = RNNMorphPredictor()
forms = predictor.predict_sentence_tags(["мама", "мыла", "раму"])
print(forms[0].pos)
>>> NOUN
print(forms[0].tag)
>>> Case=Nom|Gender=Fem|Number=Sing
print(forms[0].normal_form)
>>> мама
print(forms[0].vector)
>>> [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1]
```

### Acknowledgements ###
* Anastasyev D. G., Andrianov A. I., Indenbom E. M., 2017, [Part-of-speech Tagging with Rich Language Description](http://www.dialog-21.ru/media/3895/anastasyevdgetal.pdf), [презентация](http://www.dialog-21.ru/media/4102/anastasyev.pdf)
* [Дорожка по морфологическому анализу "Диалога-2017"](http://www.dialog-21.ru/evaluation/2017/morphology/)
* [Материалы дорожки](https://github.com/dialogue-evaluation/morphoRuEval-2017)
* [Morphine by kmike](https://github.com/kmike/morphine), [CRF classifier for MorphoRuEval-2017 by kmike](https://github.com/kmike/dialog2017)
* Tobias Horsmann and Torsten Zesch, 2017, [Do LSTMs really work so well for PoS tagging? – A replication study](http://www.ltl.uni-due.de/wp-content/uploads/horsmannZesch_emnlp2017.pdf)
* Barbara Plank, Anders Søgaard, Yoav Goldberg, 2016, [Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss](https://arxiv.org/abs/1604.05529)
