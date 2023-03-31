import os
import re
from collections import OrderedDict

from importlib_resources import files
from preprocessors.sentence_splitter import SentenceSplitter
import matplotlib.pyplot as plt
import numpy as np

basedir = str(files("resources") / "txts")
splitter = SentenceSplitter(language='hu', non_breaking_prefix_file=str(files("resources") / 'hu.txt'))


def calculate_sentence_numbers(clause_numbers: bool = False):
    sentence_lengths = {}
    clause_numbers_per_sentence = {}
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
            if length not in sentence_lengths:
                sentence_lengths[length] = 0
            sentence_lengths[length] += 1

            if length > max:
                name = file
                max = length
            if length < min:
                min = length
                name2 = file

            # clauses
            for sentence in splitted:
                clauses = len(re.split('[,;:]', sentence))
                if clauses not in clause_numbers_per_sentence:
                    clause_numbers_per_sentence[clauses] = 0
                clause_numbers_per_sentence[clauses] += 1


    print(max, name)
    print(min, name2)
    print(clause_numbers_per_sentence)


def plot_clauses():
    results = {3: 4343, 2: 6784, 1: 7741, 9: 162, 17: 34, 5: 1345, 8: 248, 4: 2422, 6: 787, 7: 460, 10: 166, 13: 58, 16: 47, 27: 12, 15: 58, 11: 82, 20: 8, 12: 90, 56: 2, 23: 15, 14: 53, 21: 16, 26: 4, 25: 5, 19: 15, 34: 3, 24: 9, 18: 16, 31: 5, 22: 6, 39: 1, 54: 1, 28: 4, 47: 2, 33: 8, 37: 3, 32: 7, 106: 2, 45: 2, 30: 4, 40: 4, 63: 1, 72: 2, 81: 3, 50: 3, 48: 1, 36: 2, 60: 1, 108: 1, 49: 1, 59: 1}
    od = OrderedDict(sorted(results.items()))

    pos = np.arange(len(results.keys()))
    width = 1.0  # gives histogram aspect to the bar diagram

    ax = plt.axes()
    # ax.set_xticks(pos + (width / 2))
    # ax.set_xticklabels(results.keys())
    value_sum = 0
    divider = 0
    for k, v in results.items():
        value_sum += k * v
        divider += v

    mean = value_sum / divider

    plt.axvline(x=mean, color='r', label='Average')
    plt.bar(results.keys(), results.values(), width, color='g')
    plt.savefig('/home/istvanu/clause_numbers.png')
    plt.show()
    print(mean)
    print(od)
    print(f"TAgmondat összesen: {value_sum}")

if __name__ == '__main__':
    # calculate_sentence_numbers(clause_numbers=True)
    plot_clauses()