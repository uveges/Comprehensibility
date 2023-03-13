from typing import List
import hu_core_news_trf


class Analyzer:

    def __init__(self):
        self.model = hu_core_news_trf.load()
        print("__________ANALYZER")

    def start(self, splitted_sentences: List[str]) -> dict[str, dict[str, List[str]]]:

        analysis_dict = {}

        for sentence in splitted_sentences:
            token_texts, lemmas, pos_tags = ([] for i in range(3))
            doc = self.model(sentence)

            for token in doc:
                token_texts.append(token.text)
                lemmas.append(token.lemma_)
                pos_tags.append(token.pos_)

            analysis_dict[sentence] = {"token_texts": token_texts,
                                       "lemmas": lemmas,
                                       "pos_tags": pos_tags}

        return analysis_dict
