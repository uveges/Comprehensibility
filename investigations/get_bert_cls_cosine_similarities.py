import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from typing import Tuple, List
from vectorizers.bert_sentence_embeddings import BertVectorizerCLS


def main():
    vectorizer = BertVectorizerCLS()

    originals, rephrased = get_list_of_sentences_by_label()

    original_similarity_matrix = get_cosine_similarities_within_group(vectorizer, originals)
    rephrased_similarity_matix = get_cosine_similarities_within_group(vectorizer, rephrased)

    inter_group_similarity = get_cosine_similarities_between_groups(vectorizer, originals, rephrased)

    print_metrics_for_similarity_matrix(original_similarity_matrix)
    print_metrics_for_similarity_matrix(rephrased_similarity_matix)


def print_metrics_for_similarity_matrix(matrix):
    ltri = np.tril(matrix, -1)                # lower triangular matrix
    ltri = ltri[np.nonzero(ltri)]
    print(f"Mean cosine similarity: {ltri.mean()}, Standard deviation: {ltri.std()}")



def get_cosine_similarities_between_groups(vectorizer, list_A, listB):
    embeddings_A = vectorizer.get_cls_token_embedding(list_A)
    embeddings_B = vectorizer.get_cls_token_embedding(listB)
    similarity_matrix = []
    for i in range(len(list_A)):
        row = []
        for j in range(len(listB)):
            # print(cosine(embeddings[i], embeddings[j]))
            row.append(cosine(list_A[i], listB[j]))
            j += 1
        similarity_matrix.append(row)
        i += 1
    # print_similarity_matrix(similarity_matrix)
    similarity_matrix = np.array(similarity_matrix)
    return similarity_matrix

def get_cosine_similarities_within_group(vectorizer, list_of_strings):
    embeddings = vectorizer.get_cls_token_embedding(list_of_strings)
    similarity_matrix = []
    for i in range(len(embeddings)):
        row = []
        for j in range(len(embeddings)):
            # print(cosine(embeddings[i], embeddings[j]))
            row.append(cosine(embeddings[i], embeddings[j]))
            j += 1
        similarity_matrix.append(row)
        i += 1
    # print_similarity_matrix(similarity_matrix)
    similarity_matrix = np.array(similarity_matrix)
    return similarity_matrix


# def print_similarity_matrix(matrix):
#     for i in range(len(matrix)):
#         print(matrix[i])

def get_list_of_sentences_by_label() -> Tuple[List[str], List[str]]:
    df = pd.read_excel("/home/istvanu/KFO_NEW/KFO_EXCEL_CORPORA/test.xlsx", engine='openpyxl')
    reprased, original = ([] for i in range(2))
    for index, row in df.iterrows():
        if row.Label == 'original':
            original.append(row.Text)
            continue
        reprased.append(row.Text)
    return original, reprased


if __name__ == '__main__':
    main()
