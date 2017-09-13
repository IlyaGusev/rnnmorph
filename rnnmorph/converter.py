class UDConverter:
    @staticmethod
    def convert_from_conllu(input_filename, output_filename, with_forth_column=False):
        with open(input_filename, "r") as r:
            with open(output_filename, "w") as w:
                for line in r:
                    if line[0] == "#" or line[0] == "=":
                        continue
                    if line != "\n":
                        records = line.split("\t")
                        if with_forth_column:
                            grammems = records[5].strip().split("|")
                        else:
                            grammems = records[4].strip().split("|")
                        dropped = ["Animacy", "Aspect", "NumType"]
                        grammems = [grammem for grammem in grammems if sum([drop in grammem for drop in dropped ]) == 0]
                        grammems = "|".join(grammems)
                        pos = records[3]
                        if pos != "PUNCT":
                            w.write("\t".join([records[1], records[2].lower(), pos, grammems]) + "\n")
                    else:
                        w.write("\n")