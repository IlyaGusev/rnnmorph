from setuptools import find_packages, setup

setup(
    name='rnnmorph',
    packages=find_packages(),
    version='0.4.1',
    description='RNNMorph: neural network disambiguation of pymorphy2 parses for precise '
                'POS-tagging in Russian language.',
    author='Ilya Gusev',
    author_email='phoenixilya@gmail.com',
    url='https://github.com/IlyaGusev/rnnmorph',
    download_url='https://github.com/IlyaGusev/rnnmorph/archive/0.4.1.tar.gz',
    keywords=['nlp', 'russian', 'lstm', 'morphology'],
    package_data={
        'rnnmorph': ['models/ru/*', 'models/en/*']
    },
    install_requires=[
        'numpy>=1.12.1',
        'scipy>=0.19.0',
        'scikit-learn>=0.18.1',
        'keras>=2.1.4',
        'h5py>=2.7.0',
        'pymorphy2>=0.8',
        'russian-tagsets==0.6',
        'tqdm>=4.14.0',
        'jsonpickle>=0.9.4',
        'nltk>=3.2.5'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',

        'Topic :: Text Processing :: Linguistic',

        'License :: OSI Approved :: Apache Software License',

        'Natural Language :: Russian',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
)
