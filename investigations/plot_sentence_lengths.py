import pandas as pd
import config
from typing import List
from nltk.tokenize import word_tokenize
from collections import OrderedDict
import matplotlib.pylab as plt
import copy


def main():
    eredeti_df = pd.read_excel(config.CORPORA_FOLDER + config.NAME_OF_ORIGINAL_CORPUS + ".xlsx")
    kozertheto_df = pd.read_excel(config.CORPORA_FOLDER + config.NAME_OF_REPHRASED_CORPUS + ".xlsx")

    token_numbers_eredeti = tokenize_sentences(eredeti_df['Text'])
    token_numbers_kozertheto = tokenize_sentences(kozertheto_df['Text'])

    plot_results(token_numbers_eredeti, config.FIGURES_FOLDER + "eredeti.png", filtering=1)
    plot_results(token_numbers_kozertheto, config.FIGURES_FOLDER + "kozertheto.png", filtering=1)


def plot_results(frequencies: OrderedDict, plot_path: str, filtering=0) -> None:
    """
    Function to plot sentence lengths in a given corpus

    :param frequencies: Ordereddict contains the frequencies to plot
    :param filtering: if not default; exclude the upper x percent of sentence lengths in order to reduce outliers
    :return:
    """

    if filtering > 0:
        total_number_of_sentences = sum(frequencies.values())
        given_percent = filtering * total_number_of_sentences / 100
        copy_frequencies = copy.deepcopy(frequencies)
        tmp_sum = 0
        for k, v in list(copy_frequencies.items())[::-1]:
            tmp_sum += v
            del copy_frequencies[k]
            if tmp_sum >= given_percent:
                x = copy_frequencies.keys()
                y = copy_frequencies.values()
                plt.bar(x, y)
                plt.savefig(plot_path)
                plt.show()
                return
    else:
        x = frequencies.keys()
        y = frequencies.values()
        plt.bar(x,y)
        plt.savefig(plot_path)
        plt.show()
        return


def tokenize_sentences(sentences: List) -> OrderedDict:
    result = {}
    for sentece in sentences:
        if len(word_tokenize(sentece)) in result:
            result[len(word_tokenize(sentece))] += 1
        else:
            result[len(word_tokenize(sentece))] = 1
    return OrderedDict(sorted(result.items()))


if __name__ == '__main__':
    main()
