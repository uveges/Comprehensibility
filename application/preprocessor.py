import re
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from importlib_resources import files
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

from preprocessors.sentence_splitter import SentenceSplitter

warnings.filterwarnings("ignore")


def prepeare_text(text):
    text = re.sub("\\r\\n", " ", text)
    text = re.sub("\\r", " ", text)
    text = re.sub("\\n", " ", text)
    text = re.sub("\\xa0", "", text)
    return text


class PreprocessContent:

    def __init__(self):
        self.splitter = SentenceSplitter(language='hu', non_breaking_prefix_file=str(files("resources") / 'hu.txt'))
        self.model = AutoModelForSequenceClassification.from_pretrained('uvegesistvan/huBERTPlain')
        self.tokenizer = AutoTokenizer.from_pretrained("uvegesistvan/huBERTPlain")
        self.trainer = Trainer(self.model)
        print("__________PREPROCESSCONTENT")

    def start(self, content: str) -> Tuple[List[str], List[str]]:
        prepeared_text = prepeare_text(content)
        sentences = self.splitter.split(prepeared_text)
        problematic_sentences = self.predict_problematic(sentences)
        return problematic_sentences, sentences

    def predict_problematic(self, list_of_sentences: List[str]) -> List[str]:
        df = pd.DataFrame(list_of_sentences, columns=["Text"])
        hg_test_data_final = Dataset.from_pandas(df)
        dataset_test_final = hg_test_data_final.map(self.tokenize_dataset)
        dataset_test_final = dataset_test_final.remove_columns(['Text'])

        test_trainer = Trainer(self.model)
        raw_pred, _, _ = test_trainer.predict(dataset_test_final)

        y_pred = np.argmax(raw_pred, axis=1)
        indices_of_problematic = np.array(np.where(y_pred == 1)).tolist()[0]
        results = [list_of_sentences[i] for i in indices_of_problematic]
        return results

    def tokenize_dataset(self, data):
        return self.tokenizer(data["Text"],
                              max_length=512,
                              truncation=True,
                              padding="max_length")
