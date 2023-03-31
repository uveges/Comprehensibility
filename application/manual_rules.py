import re
from typing import List, Tuple

import pandas as pd
from importlib_resources import files

from application.data_holder import DataHolder


# def create_light_verb_data() -> Tuple[List[str], List[str]]:
#     df = pd.read_excel(str(files("resources") / "light_verb_constructions.xlsx"))
#     to_replace = list(df.to_replace)
#     replacers = list(df.replacer)
#     return to_replace, replacers


class LexicalSuggestions:

    def __init__(self, light_verbs, light_verb_replacers, abstract_nouns, linguistic_analysis, abbreviations):
        self.light_verbs = light_verbs
        self.light_verb_replacers = light_verb_replacers
        self.abstract_nouns = abstract_nouns
        self.linguistic_analysis = linguistic_analysis
        self.law_abbreviations = abbreviations
        self.pos_not_in_sentence_length = ['PUNCT', 'SYM', 'X']
        self.negative_forms = ["ne", "nem"]
        self.threshold_for_long_sentence = 35
        self.law_reference_pat = re.compile("(?#törvény)([0-9]{4}\.\s?évi\s?M{0,3}(CM|CD|D?C{0,3})?(XC|XL|L?X{0,3})?(IX|IV|V?I{0,3})?\.\s?(törvény)?)(?#kormányrendelet)|[0-9]{,3}/[12][0-9]{3}\.?\s?\((CM|CD|D?C{0,3})?(XC|XL|L?X{0,3})?(IX|IV|V?I{0,3})\.\s?[0-9]{1,2}\.\)\s?(Korm(ány|\.)\s?rendelet)", re.VERBOSE)

    def start(self, suggestions_per_sentence):
        for sentence, analysis in self.linguistic_analysis.items():
            if isinstance(suggestions_per_sentence[sentence], str) and suggestions_per_sentence[sentence] == "OK!":
                continue

            # Search for light verbs
            for index, lv in enumerate(self.light_verbs):
                if lv in sentence:
                    suggestions_per_sentence[sentence]['light_verbs'].append(
                        f"{lv} -> {self.light_verb_replacers[index],}")
            for abbreviation in self.law_abbreviations:
                if abbreviation in sentence:
                    suggestions_per_sentence[sentence]['abbreviations'].append(abbreviation)
            # calculate abstract nouns' proportion
            if self._is_abstract(sentence):
                suggestions_per_sentence[sentence]['too_abstract'] = True
            if self._is_verbless(sentence):
                suggestions_per_sentence[sentence]['verbless'] = True
            if self._is_too_long(sentence):
                suggestions_per_sentence[sentence]['too_long'] = True
            law_references = re.findall(self.law_reference_pat, sentence)
            if law_references:
                for index, ref in enumerate(law_references):
                    suggestions_per_sentence[sentence]['references'].append(ref[index])
            if len(self.linguistic_analysis[sentence]["clauses"]) > 10:
                suggestions_per_sentence[sentence]['too_much_clauses'] = True
            if self.multiple_negations(sentence):
                suggestions_per_sentence[sentence]['multiple_negations'] = True

    def multiple_negations(self, sentence: str):
        """
        Checks if any two consecutive clauses contains negative forms.
        :param sentence:
        :return:
        """

        clauses = self.linguistic_analysis[sentence]["clauses"]
        for i, c in enumerate(clauses):
            if i == len(clauses):
                continue
            if any([x in c for x in self.negative_forms]) and any([x in clauses[i+1] for x in self.negative_forms]):
                return True
        return False

    def _is_too_long(self, sentence: str) -> bool:
        important_pos_tags = [pos for pos in self.linguistic_analysis[sentence]["pos_tags"] if pos not in self.pos_not_in_sentence_length]
        if len(important_pos_tags) >= self.threshold_for_long_sentence:
            return True
        return False

    def _is_verbless(self, sentence: str) -> bool:
        for pos in self.linguistic_analysis[sentence]["pos_tags"]:
            if pos == "VERB":
                return False
        return True

    def _is_abstract(self, sentence: str) -> bool:
        pos_tags = self.linguistic_analysis[sentence]["pos_tags"]
        noun_indices = []
        for index, pos in enumerate(pos_tags):
            if pos == "NOUN":
                noun_indices.append(index)
        abstract_count = 0
        for index in noun_indices:
            if self.linguistic_analysis[sentence]['lemmas'][index] in self.abstract_nouns:
                abstract_count += 1
        threshold = len(noun_indices) * 0.9
        return True if abstract_count >= threshold else False

# if __name__ == '__main__':
#     changer = LexicalChanges()
#     print(changer.start())
