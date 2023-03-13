import os
import re

from importlib_resources import files
from preprocessors.sentence_splitter import SentenceSplitter
import matplotlib.pyplot as plt

basedir = str(files("resources") / "txts")


if __name__ == '__main__':

    splitter = SentenceSplitter(language='hu', non_breaking_prefix_file=str(files("resources") / 'hu.txt'))
    lengths = {}
    max = 0
    min = 10000
    name = ""
    name2 = ""

    for file in os.listdir(basedir):
        full_path = os.path.join(basedir, file)
        with open(full_path, 'r', encoding='utf8') as input:
            content = " ".join(input.readlines())
            content = re.sub("\\r\\n", " ", content)
            content = re.sub("\\r", " ", content)
            content = re.sub("\\n", " ", content)

            splitted = splitter.split(content)
            length = len(splitted)
            if not length in lengths:
                lengths[length] = 0
            lengths[length] += 1

            if length > max:
                name = file
                max = length
            if length < min:
                min = length
                name2 = file

    print(max, name)
    print(min, name2)