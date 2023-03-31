import re
import warnings
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from importlib_resources import files
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from preprocessors.sentence_splitter import SentenceSplitter
from vectorizers.bert_sentence_embeddings import BertVectorizerCLS

warnings.filterwarnings('ignore')


def prepeare_text(text):
    text = re.sub("\\r\\n", " ", text)
    text = re.sub("\\r", " ", text)
    text = re.sub("\\n", " ", text)
    text = re.sub("\\xa0", "", text)
    return text


def define_min_max_cluster_number(t: int) -> Tuple[int, int]:
    """

    :param t: Number of sentences in the given text
    :return: Tuple: min, max number of clusters to investigate
    """

    target_upper = 10  # d
    target_lower = 5  # c
    real_upper = 641  # b
    real_lower = 100  # a

    if t <= 5:
        return 1, 1
    elif 6 <= t <= 100:
        return 3, 5
    else:
        return 3, round(target_lower + ((target_upper - target_lower) / (real_upper - real_lower) * (t - real_lower)))


def _print_clustering_result(df: pd.DataFrame) -> None:

    plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b')
    plt.xlabel('K')
    plt.ylabel('Gap Statistic')
    plt.title('Gap Statistic vs. K')
    plt.show()


class ClusterComposer:

    def __init__(self, text: str):
        self.vectorizer = BertVectorizerCLS()
        self.splitter = SentenceSplitter(language='hu', non_breaking_prefix_file=str(files("resources") / 'hu.txt'))
        self.text = prepeare_text(text)
        self.splitted = []
        self.CLS_embeddings = None
        self.kmeans_kwargs = {"max_iter": 500, "random_state": 42, "init": "random"}
        self.cluster_center_vectors = None  # Centers of clusters, NOT vectors from train data!
        self.max_clusters = None
        self.optimal_cluster_number = None
        self.sentence_per_cluster_to_summary = 1
        self.optimal_kmeans_model = None

    def start(self, draw_clustering_optimization: bool = False) -> List[str]:
        """
        Determines the optimal number of clusters, then fits a KMeans model with this optimal number
        :return: List of sentences that are closest to these clusters' centroids
        """
        self._vectorize_data()
        optimal_clusters_number, df = self._define_optimal_number_of_clusters(self.CLS_embeddings, nrefs=5)
        self.optimal_cluster_number = optimal_clusters_number

        if draw_clustering_optimization:
            _print_clustering_result(df)

        self._fit_kmeans_optimal(optimal_clusters_number)
        center_sentences = self._find_cluster_center_sentences()

        self._test_clusters()

        return center_sentences

    def _test_clusters(self):
        labels = self.optimal_kmeans_model.predict(self.CLS_embeddings)
        unique_labels = sorted(set(labels))
        predictions = {}
        for label in unique_labels:
            predictions[label] = []
        for label, sentence in zip(labels, self.splitted):
            predictions[label].append(sentence)

        # print(predictions)

    def _find_cluster_center_sentences(self) -> List[str]:

        # ismerjük a klaszter középpontokat: self..cluster_center_vectors
        # mérjük meg az összes mondat vektorának (self.CLS_embeddings) távolságát a középpontoktól
        # az egyes középpontokhoz legközelebb eső x darab mondat vektorának indexét keressük meg a self.CLS_embeddings -ben
        # az ezen az indexen levő mondatokat adjuk vissza a self.splitted -ből (indexelése egyezik a self.CLS_embeddings-ével

        distances = pairwise_distances(self.cluster_center_vectors, self.CLS_embeddings, metric='euclidean')
        ind = [np.argpartition(i, 1)[:self.sentence_per_cluster_to_summary] for i in distances]
        sentences = [self.splitted[indexes[0].astype(int)] for indexes in ind]
        return sentences

    def _fit_kmeans_optimal(self, k: int) -> None:
        """
        Fits the model with optimal cluster number, and saves cluster centroids to self.cluster_centers
        :param k: number of clusters
        :return: None
        """

        model = KMeans(n_clusters=k, **self.kmeans_kwargs).fit(self.CLS_embeddings)
        self.optimal_kmeans_model = model
        self.cluster_center_vectors = model.cluster_centers_

    def _define_optimal_number_of_clusters(self, data: pd.DataFrame, nrefs: int = 3) -> Tuple[int, pd.DataFrame]:
        """
        Calculates KMeans optimal K using Gap Statistic

        :param data: previously vectorized dataset
        :param nrefs: reference number
        :param maxClusters: maximum number of potential clusters
        :return:
        """
        self.min_clusters, self.max_clusters = define_min_max_cluster_number(len(self.splitted))

        gaps = np.zeros(self.max_clusters - self.min_clusters + 1)
        # print(self.min_clusters, self.max_clusters)
        resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})  # Holder for reference dispersion results

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for gap_index, k in tqdm(enumerate(range(self.min_clusters, self.max_clusters + 1))):
            refDisps = np.zeros(nrefs)
            for i in range(nrefs):
                randomReference = np.random.random_sample(size=data.shape)  # Create new random reference set

                km = KMeans(k, **self.kmeans_kwargs)  # Fitting the model
                km.fit(randomReference)

                refDisp = km.inertia_
                refDisps[i] = refDisp  # Fit cluster to original data and create dispersion
                km = KMeans(k, **self.kmeans_kwargs)
                km.fit(data)

                origDisp = km.inertia_  # Calculate gap statistic
                gap = np.log(np.mean(refDisps)) - np.log(origDisp)  # Assign this loop's gap statistic to gaps
                gaps[gap_index] = gap

            resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

        # print(gaps)
        return gaps.argmax() + self.min_clusters, resultsdf

    def _vectorize_data(self):
        splitted = self.splitter.split(self.text)
        while "" in splitted:
            splitted.remove("")
        splitted = [sentence.replace("\\t", "").replace("\t", " ") for sentence in splitted]
        self.splitted = splitted
        self.CLS_embeddings = self.vectorizer.get_cls_token_embedding(self.splitted)
