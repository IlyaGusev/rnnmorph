# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Конвертер всех корпусов к одному формату.

from rnnmorph.data_preparation.process_tag import process_gram_tag


class UDConverter:
    @staticmethod
    def convert_from_conllu(input_filename, output_filename, with_forth_column=False, with_punct=True):
        with open(input_filename, "r", encoding='utf-8') as r, open(output_filename, "w", encoding='utf-8') as w:
            for line in r:
                if line[0] == "#" or line[0] == "=":
                    continue
                if line == "\n":
                    w.write("\n")
                    continue
                records = line.split("\t")
                pos = records[3]
                if with_forth_column:
                    gram = records[5]
                else:
                    gram = records[4]
                gram = process_gram_tag(gram)
                if pos == "PUNCT" and not with_punct:
                    continue
                w.write("\t".join([records[1], records[2].lower(), pos, gram]) + "\n")