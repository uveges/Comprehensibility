import hu_core_news_lg
import os
from pathlib import Path
from typing import List
from nltk.tokenize import word_tokenize


def compare_gold_standard_and_segmented(gold_standard: list, segmented: list) -> List:
    """

    :param gold_standard:
    :param segmented:
    :return: List: correctly segmented sentences, incorrect segmented sentences,
    incorrectly segmented original sentences, metrics: correct count, correct rate
    """

    # először nltk-val tokenizáljuk minda kettőt, majd a kapott elemeket összevetjük

    tokenized_gold_standard_sentences, tokenized_segmented_sentences = ([] for i in range(2))
    for gold_standard_sentence in gold_standard:
        tokenized_gold_standard_sentences.append(word_tokenize(gold_standard_sentence))
    for segmented_sentence in segmented:
        tokenized_segmented_sentences.append(word_tokenize(segmented_sentence))

    # lists are mutable, they cannot be hashed. The best bet is to convert them to a tuple and form a set
    # set(tuple(row) for row in mat)
    tokenized_gold_standard_sentences = set(tuple(x) for x in tokenized_gold_standard_sentences)
    tokenized_segmented_sentences = set(tuple(x) for x in tokenized_segmented_sentences)

    common = tokenized_gold_standard_sentences.intersection(tokenized_segmented_sentences)
    only_in_segmented = tokenized_segmented_sentences.difference(
        tokenized_gold_standard_sentences)  # hibás szegmentáltak
    not_in_segmented = tokenized_gold_standard_sentences.difference(
        tokenized_segmented_sentences)  # hibásan szegmentáltak

    count_correctly_segmented = len(common)
    rate_correctly_segmented = round(len(common) / len(tokenized_gold_standard_sentences), 2)
    metrics = [count_correctly_segmented, rate_correctly_segmented]

    return [list(common), list(only_in_segmented), list(not_in_segmented), metrics]


def segment_with_spacy(string_to_process, nlp):
    doc = nlp(string_to_process)
    results = []
    tmp = []
    for sent in list(doc.sents):
        for token in sent:
            tmp.append(token.text)
        results.append(' '.join(tmp))
        tmp.clear()
    return results


def read_content(file: str) -> List:
    with open(file, 'r', encoding='utf8') as input:
        return input.readlines()


def create_metrics(base_dir, file, nlp):
    gold_standard = [x.replace('\n', '') for x in read_content(os.path.join(base_dir, file))]
    gold_standard_str = ''.join(gold_standard)
    segmented = segment_with_spacy(gold_standard_str, nlp)
    return compare_gold_standard_and_segmented(gold_standard, segmented)


def write_results(results, base_dir, sub_corpus_name):
    result_file = Path(base_dir) / f"{sub_corpus_name}_hu_core_news_lg_results.txt"
    with open(result_file, 'w', encoding='utf8') as output:
        # metrics
        output.write(f"Correctly segmented count: {str(results[3][0])}\n")
        output.write(f"Correctly segmented rate: {str(results[3][1])}\n")
        output.write("\n_________Correct sentences:________________________________\n\n")
        for sent in results[0]:
            output.write(' '.join(sent))
            output.write('\n')
        output.write("\n_________Incorrect sentences in segmented list:____________\n\n")
        for sent in results[1]:
            output.write(' '.join(sent))
            output.write('\n')
        output.write("\n_________Original sentences incorrectly segmented:_________\n\n")
        for sent in results[2]:
            output.write(' '.join(sent))
            output.write('\n')


class ValiateSpacyModel():
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def validate(self):
        nlp = hu_core_news_lg.load()

        metrics = {}  # metrics: correct count, correct rate

        for file in os.listdir(self.base_dir):
            if file.endswith("hatarozat.txt"):
                results = create_metrics(self.base_dir, file, nlp)
                write_results(results, self.base_dir, "hatarozat")
                metrics["hatarozat"] = results[3]
            if file.endswith("nav_tajekoztato.txt"):
                results = create_metrics(self.base_dir, file, nlp)
                write_results(results, self.base_dir, "nav_tajekoztato")
                metrics["nav_tajekoztato"] = results[3]
            if file.endswith("torveny.txt"):
                results = create_metrics(self.base_dir, file, nlp)
                write_results(results, self.base_dir, "torveny")
                metrics["torveny"] = results[3]
            if file.endswith("ujsaghir.txt"):
                results = create_metrics(self.base_dir, file, nlp)
                write_results(results, self.base_dir, "ujsaghir")
                metrics["ujsaghir"] = results[3]


        print("hu_core_news_lg:")
        print(metrics)