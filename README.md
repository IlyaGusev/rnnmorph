# rnnmorph
[![Current version on PyPI](http://img.shields.io/pypi/v/rnnmorph.svg)](https://pypi.python.org/pypi/rnnmorph)
[![Python versions](https://img.shields.io/pypi/pyversions/rnnmorph.svg)](https://pypi.python.org/pypi/rnnmorph)
[![Code Climate](https://codeclimate.com/github/IlyaGusev/rnnmorph/badges/gpa.svg)](https://codeclimate.com/github/IlyaGusev/rnnmorph)

Морфологический анализатор на основе нейронных сетей и pymorphy2.

Lenta:
* Качество по тегам:
  * 4000 меток из 4179, точность 95.72%
  * 260 предложений из 358, точность 72.63%
* Качество полного разбора:
  * 3868 слов из 4179, точность 92.56%
  * 179 предложений из 358, точность 50.00%

VK:
* Качество по тегам:
  * 3680 меток из 3877, точность 94.92%
  * 422 предложений из 568, точность 74.30%
* Качество полного разбора:
  * 3557 слов из 3877, точность 91.75%
  * 344 предложений из 568, точность 60.56%

JZ:
* Качество по тегам:
  * 3873 меток из 4042, точность 95.82%
  * 282 предложений из 394, точность 71.57%
* Качество полного разбора:
  * 3655 слов из 4042, точность 90.43%
  * 169 предложений из 394, точность 42.89%


### На основе: ###
* Anastasyev D. G., Andrianov A. I., Indenbom E. M., 2017, [Part-of-speech Tagging with Rich Language Description](http://www.dialog-21.ru/media/3895/anastasyevdgetal.pdf), [презентация](http://www.dialog-21.ru/media/4102/anastasyev.pdf)
* [Дорожка по морфологическому анализу "Диалога-2017"](http://www.dialog-21.ru/evaluation/2017/morphology/)
* [Материалы дорожки](https://github.com/dialogue-evaluation/morphoRuEval-2017)

