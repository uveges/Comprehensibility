from typing import List
import numpy as np


class DataHolder:

    def __init__(self):
        self._light_verbs = []
        self._light_verb_replacers = []
        self._abstract_nouns = []
        self._splitted_sentences_from_text = []
        self._problematic_sentences_from_splitted = []
        self._sentence_vectors = None
        self._linguistic_analysis = None
        self._law_abbreviations = []

    @property
    def light_verbs(self) -> List[str]:
        return self._light_verbs

    @property
    def light_verb_replacers(self) -> List[str]:
        return self._light_verb_replacers

    @property
    def abstract_nouns(self) -> List[str]:
        return self._abstract_nouns

    @property
    def splitted_sentences_from_text(self) -> List[str]:
        return self._splitted_sentences_from_text

    @property
    def sentence_vectors(self) -> np.array:
        return self._sentence_vectors

    @property
    def linguistic_analysis(self) -> dict[str, dict[str, List[str]]]:
        return self._linguistic_analysis

    @property
    def problematic_sentences_from_splitted(self) -> List[str]:
        return self._problematic_sentences_from_splitted

    @property
    def law_abbreviations(self) -> List[str]:
        return self._law_abbreviations

    @light_verbs.setter
    def light_verbs(self, list: List[str]):
        if not self._light_verbs:
            self._light_verbs = list
        else:
            raise UserWarning("Function verbs were already stored, but now changed!")

    @light_verb_replacers.setter
    def light_verb_replacers(self, list: List[str]):
        if not self._light_verb_replacers:
            self._light_verb_replacers = list
        else:
            raise UserWarning("Function verb replacers were already stored, but now changed!")

    @abstract_nouns.setter
    def abstract_nouns(self, list: List[str]):
        if not self._abstract_nouns:
            self._abstract_nouns = list
        else:
            raise UserWarning("Abstract nouns were already stored, but now changed!")

    @splitted_sentences_from_text.setter
    def splitted_sentences_from_text(self, list: List[str]):
        if not self._splitted_sentences_from_text:
            self._splitted_sentences_from_text = list
        else:
            raise UserWarning("Splitted sentences were already stored, but now changed!")

    @sentence_vectors.setter
    def sentence_vectors(self, array: np.array):
        if not self._sentence_vectors:
            self._sentence_vectors = array
        else:
            raise UserWarning("Sentence vectors were already stored, but now changed!")


    @linguistic_analysis.setter
    def linguistic_analysis(self, analysis: dict[str, dict[str, List[str]]]):
        if not self._linguistic_analysis:
            self._linguistic_analysis = analysis
        else:
            raise UserWarning("Linguistic analysis was already stored, but now changed!")

    @problematic_sentences_from_splitted.setter
    def problematic_sentences_from_splitted(self, problematic_sentence_list: List[str]):
        if not self._problematic_sentences_from_splitted:
            self._problematic_sentences_from_splitted = problematic_sentence_list
        else:
            raise UserWarning("Problematic sentences were already stored, but now changed!")

    @law_abbreviations.setter
    def law_abbreviations(self, abbreviations: List[str]):
        if not self._law_abbreviations:
            self._law_abbreviations = abbreviations
        else:
            raise UserWarning("Law abbreviations were alreaday stored, but now changed!")