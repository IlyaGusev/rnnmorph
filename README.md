# rnnmorph
[![Current version on PyPI](http://img.shields.io/pypi/v/rnnmorph.svg)](https://pypi.python.org/pypi/rnnmorph)
[![Python versions](https://img.shields.io/pypi/pyversions/rnnmorph.svg)](https://pypi.python.org/pypi/rnnmorph)
[![Build Status](https://travis-ci.org/IlyaGusev/rnnmorph.svg?branch=master)](https://travis-ci.org/IlyaGusev/rnnmorph)
[![Code Climate](https://codeclimate.com/github/IlyaGusev/rnnmorph/badges/gpa.svg)](https://codeclimate.com/github/IlyaGusev/rnnmorph)

Морфологический анализатор на основе нейронных сетей и pymorphy2.

Lenta:
* Качество по тегам:
  * 4006 меток из 4179, точность 95.86%
  * 265 предложений из 358, точность 74.02%
* Качество полного разбора:
  * 3867 слов из 4179, точность 92.53%
  * 185 предложений из 358, точность 51.68%

VK:
* Качество по тегам:
  * 3612 меток из 3877, точность 93.16%
  * 383 предложений из 568, точность 67.43%
* Качество полного разбора:
  * 3493 слов из 3877, точность 90.10%
  * 311 предложений из 568, точность 54.75%

JZ:
* Качество по тегам:
  * 3853 меток из 4042, точность 95.32%
  * 280 предложений из 394, точность 71.07%
* Качество полного разбора:
  * 3638 слов из 4042, точность 90.00%
  * 169 предложений из 394, точность 42.89%
  
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
```

### Acknowledgements ###
* Anastasyev D. G., Andrianov A. I., Indenbom E. M., 2017, [Part-of-speech Tagging with Rich Language Description](http://www.dialog-21.ru/media/3895/anastasyevdgetal.pdf), [презентация](http://www.dialog-21.ru/media/4102/anastasyev.pdf)
* [Дорожка по морфологическому анализу "Диалога-2017"](http://www.dialog-21.ru/evaluation/2017/morphology/)
* [Материалы дорожки](https://github.com/dialogue-evaluation/morphoRuEval-2017)
* [Morphine by kmike](https://github.com/kmike/morphine), [CRF classifier for MorphoRuEval-2017 by kmike](https://github.com/kmike/dialog2017)
* Tobias Horsmann and Torsten Zesch, 2017, [Do LSTMs really work so well for PoS tagging? – A replication study](http://www.ltl.uni-due.de/wp-content/uploads/horsmannZesch_emnlp2017.pdf)
* Barbara Plank, Anders Søgaard, Yoav Goldberg, 2016, [Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss](https://arxiv.org/abs/1604.05529)
