from typing import List
import hu_core_news_trf
import re


class Analyzer:

    def __init__(self):
        self.model = hu_core_news_trf.load()
        self.doc = None
        print("__________ANALYZER")

    def start(self, splitted_sentences: List[str]) -> dict[str, dict[str, List[str]]]:

        analysis_dict = {}

        for sentence in splitted_sentences:
            token_texts, lemmas, pos_tags = ([] for i in range(3))
            self.doc = self.model(sentence)

            for token in self.doc:
                token_texts.append(token.text)
                lemmas.append(token.lemma_)
                pos_tags.append(token.pos_)

            clauses = re.split('[,;:]', sentence)

            analysis_dict[sentence] = {"token_texts": token_texts,
                                       "lemmas": lemmas,
                                       "pos_tags": pos_tags,
                                       "clauses": clauses
                                       }

        return analysis_dict

