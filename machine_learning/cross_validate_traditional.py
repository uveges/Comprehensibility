import json
import os
import warnings
from statistics import mean, pstdev
from typing import List, Union

import hu_core_news_trf
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

import config

warnings.filterwarnings('ignore')


def lemmatize_full(texts) -> List:
    nlp = hu_core_news_trf.load()
    lemmatizedList = list()

    for index, row in tqdm(texts.iteritems()):
        lemmatizedList.append(lemmatize(row, nlp))

    return lemmatizedList


def lemmatize(text: str, nlp) -> str:
    doc = nlp(text)
    returnlist = list()
    for token in doc:
        returnlist.append(token.lemma_)
    return " ".join(returnlist)


class CrossValidation:

    def __init__(self, all_data, preprocess):
        self.all_data = all_data
        self.directory = os.path.dirname(all_data)
        self.basename = os.path.basename(all_data)
        self.random_state = 42
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
        self.preprocess = preprocess

    def start(self):

        df = pd.read_excel(self.all_data)
        texts = df.Text
        labels = df.Label

        expected_lemmatized = os.path.join(self.directory, self.basename.replace(".xlsx", "_lemmatized.xlsx"))
        print(f"Searching for lemmatized data: {expected_lemmatized}")
        if os.path.exists(expected_lemmatized):
            print(f"Found, finding optimal model...")
            self.start_optimization(expected_lemmatized)
        else:
            print("Not found, generating lemmatized version from data...")
            texts_lemmatized_list = lemmatize_full(texts)
            df_lemm = pd.DataFrame(columns=['Text', 'Label'])
            df_lemm['Text'] = texts_lemmatized_list
            df_lemm['Label'] = labels
            df_lemm.to_excel(expected_lemmatized)
            self.start_optimization(expected_lemmatized)

    def start_optimization(self, lemmatized_dataframe: Union[str, bytes]):
        df = pd.read_excel(lemmatized_dataframe)
        texts = df.Text
        labels = df.Label

        if self.preprocess:
            texts = self.start_preprocess(texts)

        train_text, test_texts, train_labels, test_labels = train_test_split(texts, labels, train_size=0.9,
                                                                             random_state=self.random_state)
        self.start_SVM_optimization(train_text, train_labels, k=10)
        self.start_LR_optimization(train_text, train_labels, k=10)

    def start_preprocess(self, texts: pd.Series) -> pd.Series:

        texts = texts.str.lower()
        texts = texts.str.replace('\d+', '')
        texts = texts.str.replace(r'[^\w\s]+', '')
        texts = texts.apply(word_tokenize)
        texts = texts.apply(lambda words: [word for word in words if word not in config.STOPWORDS])
        texts = texts.str.join(" ")

        return texts

    def start_SVM_optimization(self, train_texts: pd.Series, train_labels: pd.Series, k: int = 10) -> None:
        """Performs k-fold cross validation with 4^3 parameter combinations

        :param train_texts: Series of train examples to be split in a k-fold way
        :param train_labels: Series of corresponding train labels
        :param k: The number of splits for train / validation selection
        :return:
        """
        print("Support Vector Machines started...")

        Cs = [0.1, 1, 10, 100]
        gammas = [1, 0.1, 0.01, 0.001]
        kernels = ['rbf', 'poly', 'sigmoid', 'linear']

        k_fold = len(train_texts) / k
        results_per_parameter_set = {'AVGs': {}, 'STDs': {}}

        total_combinations = len(Cs) * len(gammas) * len(kernels)

        for index_c, C in enumerate(Cs):
            for index_g, Gamma in enumerate(gammas):
                for index_k, Kernel in enumerate(kernels):
                    print(f"{index_k+index_g+index_c} / {total_combinations}")

                    # store values for cross validation
                    parameters = str(C) + " " + str(Gamma) + " " + str(Kernel)
                    results_per_parameter_set[parameters] = {
                        'original_P': [],
                        'original_R': [],
                        'original_F': [],
                        'rephrased_P': [],
                        'rephrased_R': [],
                        'rephrased_F': [],
                    }

                    for i in range(k):
                        begin_test = int(i * k_fold)
                        end_test = int((i + 1) * k_fold)
                        print(f"Cross -validation step: {i + 1}")

                        # The test set will be one tenth of the original
                        X_test = train_texts[begin_test: end_test]
                        y_test = train_labels[begin_test: end_test]

                        # We are getting the train  set by dropping this tenth from the original data
                        X_train = train_texts.drop(train_texts.index[begin_test: end_test], axis=0)
                        y_train = train_labels.drop(train_labels.index[begin_test: end_test], axis=0)

                        self.vectorizer.fit(X_train)
                        X_train = self.vectorizer.transform(X_train)
                        X_test = self.vectorizer.transform(X_test)

                        SVCmodel = SVC(C=C, gamma=Gamma, kernel=Kernel, class_weight='balanced')
                        SVCmodel.fit(X_train, y_train)
                        y_pred = SVCmodel.predict(X_test)

                        report = classification_report(y_true=y_test,
                                                       y_pred=y_pred,
                                                       target_names=['original', 'rephrased'],
                                                       digits=4,
                                                       output_dict=True,
                                                       labels=['original', 'rephrased'])

                        results_per_parameter_set[parameters]['original_P'].append(report['original']['precision'])
                        results_per_parameter_set[parameters]['original_R'].append(report['original']['recall'])
                        results_per_parameter_set[parameters]['original_F'].append(report['original']['f1-score'])
                        results_per_parameter_set[parameters]['rephrased_P'].append(report['rephrased']['precision'])
                        results_per_parameter_set[parameters]['rephrased_R'].append(report['rephrased']['recall'])
                        results_per_parameter_set[parameters]['rephrased_F'].append(report['rephrased']['f1-score'])

                    results_per_parameter_set['AVGs'][parameters] = {}
                    results_per_parameter_set['AVGs'][parameters]['original_P'] = mean(results_per_parameter_set[parameters]['original_P'])
                    results_per_parameter_set['AVGs'][parameters]['original_R'] = mean(results_per_parameter_set[parameters]['original_R'])
                    results_per_parameter_set['AVGs'][parameters]['original_F'] = mean(results_per_parameter_set[parameters]['original_F'])
                    results_per_parameter_set['AVGs'][parameters]['rephrased_P'] = mean(results_per_parameter_set[parameters]['rephrased_P'])
                    results_per_parameter_set['AVGs'][parameters]['rephrased_R'] = mean(results_per_parameter_set[parameters]['rephrased_R'])
                    results_per_parameter_set['AVGs'][parameters]['rephrased_F'] = mean(results_per_parameter_set[parameters]['rephrased_F'])

                    results_per_parameter_set['STDs'][parameters] = {}
                    results_per_parameter_set['STDs'][parameters]['original_P'] = pstdev(results_per_parameter_set[parameters]['original_P'])
                    results_per_parameter_set['STDs'][parameters]['original_R'] = pstdev(results_per_parameter_set[parameters]['original_R'])
                    results_per_parameter_set['STDs'][parameters]['original_F'] = pstdev(results_per_parameter_set[parameters]['original_F'])
                    results_per_parameter_set['STDs'][parameters]['rephrased_P'] = pstdev(results_per_parameter_set[parameters]['rephrased_P'])
                    results_per_parameter_set['STDs'][parameters]['rephrased_R'] = pstdev(results_per_parameter_set[parameters]['rephrased_R'])
                    results_per_parameter_set['STDs'][parameters]['rephrased_F'] = pstdev(results_per_parameter_set[parameters]['rephrased_F'])

        to_write = json.dumps(results_per_parameter_set)
        with open(os.path.join(self.directory, "SVM_results.json"), 'w') as output:
            output.write(to_write)

        print(f"Results saved: {os.path.join(self.directory, 'SVM_results.json')}")

    def start_LR_optimization(self, train_texts: pd.Series, train_labels: pd.Series, k: int = 10) -> None:
        """Performs k-fold cross validation with 5^2 parameter combinations

        :param train_texts: Series of train examples to be split in a k-fold way
        :param train_labels: Series of corresponding train labels
        :param k: The number of splits for train / validation selection
        :return:
        """
        print("Logistic Regressions started...")

        solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        c_values = [1000, 100, 10, 1.0, 0.1, 0.01]

        k_fold = len(train_texts) / k
        results_per_parameter_set = {'AVGs': {}, 'STDs': {}}

        total_combinations = len(solvers) * len(c_values)

        for index_c, C_value in enumerate(c_values):
            for index_s, Solver in enumerate(solvers):
                print(f"{index_s + index_c} / {total_combinations}")

                # store values for cross validation
                parameters = str(C_value) + " " + str(Solver)
                results_per_parameter_set[parameters] = {
                    'original_P': [],
                    'original_R': [],
                    'original_F': [],
                    'rephrased_P': [],
                    'rephrased_R': [],
                    'rephrased_F': [],
                }

                for i in range(k):
                    print(f"Cross-validation step: {i + 1}")
                    begin_test = int(i * k_fold)
                    end_test = int((i + 1) * k_fold)

                    # The test set will be one tenth of the original
                    X_test = train_texts[begin_test: end_test]
                    y_test = train_labels[begin_test: end_test]

                    # We are getting the train  set by dropping this tenth from the original data
                    X_train = train_texts.drop(train_texts.index[begin_test: end_test], axis=0)
                    y_train = train_labels.drop(train_labels.index[begin_test: end_test], axis=0)

                    self.vectorizer.fit(X_train)
                    X_train = self.vectorizer.transform(X_train)
                    X_test = self.vectorizer.transform(X_test)

                    LRmodel = LogisticRegression(C=C_value, solver=Solver, max_iter=1000)
                    LRmodel.fit(X_train, y_train)
                    y_pred = LRmodel.predict(X_test)

                    report = classification_report(y_true=y_test,
                                                   y_pred=y_pred,
                                                   target_names=['original', 'rephrased'],
                                                   digits=4,
                                                   output_dict=True,
                                                   labels=['original', 'rephrased'])

                    results_per_parameter_set[parameters]['original_P'].append(report['original']['precision'])
                    results_per_parameter_set[parameters]['original_R'].append(report['original']['recall'])
                    results_per_parameter_set[parameters]['original_F'].append(report['original']['f1-score'])
                    results_per_parameter_set[parameters]['rephrased_P'].append(report['rephrased']['precision'])
                    results_per_parameter_set[parameters]['rephrased_R'].append(report['rephrased']['recall'])
                    results_per_parameter_set[parameters]['rephrased_F'].append(report['rephrased']['f1-score'])

                results_per_parameter_set['AVGs'][parameters] = {}
                results_per_parameter_set['AVGs'][parameters]['original_P'] = mean(
                    results_per_parameter_set[parameters]['original_P'])
                results_per_parameter_set['AVGs'][parameters]['original_R'] = mean(
                    results_per_parameter_set[parameters]['original_R'])
                results_per_parameter_set['AVGs'][parameters]['original_F'] = mean(
                    results_per_parameter_set[parameters]['original_F'])
                results_per_parameter_set['AVGs'][parameters]['rephrased_P'] = mean(
                    results_per_parameter_set[parameters]['rephrased_P'])
                results_per_parameter_set['AVGs'][parameters]['rephrased_R'] = mean(
                    results_per_parameter_set[parameters]['rephrased_R'])
                results_per_parameter_set['AVGs'][parameters]['rephrased_F'] = mean(
                    results_per_parameter_set[parameters]['rephrased_F'])

                results_per_parameter_set['STDs'][parameters] = {}
                results_per_parameter_set['STDs'][parameters]['original_P'] = pstdev(
                    results_per_parameter_set[parameters]['original_P'])
                results_per_parameter_set['STDs'][parameters]['original_R'] = pstdev(
                    results_per_parameter_set[parameters]['original_R'])
                results_per_parameter_set['STDs'][parameters]['original_F'] = pstdev(
                    results_per_parameter_set[parameters]['original_F'])
                results_per_parameter_set['STDs'][parameters]['rephrased_P'] = pstdev(
                    results_per_parameter_set[parameters]['rephrased_P'])
                results_per_parameter_set['STDs'][parameters]['rephrased_R'] = pstdev(
                    results_per_parameter_set[parameters]['rephrased_R'])
                results_per_parameter_set['STDs'][parameters]['rephrased_F'] = pstdev(
                    results_per_parameter_set[parameters]['rephrased_F'])

        to_write = json.dumps(results_per_parameter_set)
        with open(os.path.join(self.directory, "LR_results.json"), 'w') as output:
            output.write(to_write)

        print(f"Results saved: {os.path.join(self.directory, 'LR_results.json')}")
