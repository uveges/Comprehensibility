from application.clustering import ClusterComposer
from importlib_resources import files
from application.manual_rules import LexicalSuggestions
from application.linguistic_analyzer import Analyzer
from application.preprocessor import PreprocessContent
from application.data_holder import DataHolder
import pandas as pd
from typing import List

from fastapi import FastAPI
from typing import Dict, Union, Optional
from pydantic import BaseModel
import json

from docs.export_project_docs import markdown_exporter


# def main(test_text):

class Document(BaseModel):
    content: str


app = FastAPI(title="AccessibleAPI",
              description=markdown_exporter.export_md_files_as_text())  # callable: uvicorn application.main:app


@app.get('/')
def root():
    return {"Response": "App is working, see http://127.0.0.1:8000/docs for testing!"}


@app.post("/document", response_model=Dict[str, Union[str, List, bool]])
def process_document(document: Document):  # read request body as JSON
    response = document.dict()
    original_content = response["content"]

    holder = store_data_in_data_holder_object(original_content)

    # print(holder.linguistic_analysis)

    response["problematic"] = str(holder.problematic_sentences_from_splitted)

    # creating dict to store all modification suggestions
    changes_per_sentence = init_dict_changes(holder.splitted_sentences_from_text,
                                             holder.problematic_sentences_from_splitted)

    # NOT REPLACING! JUST SUGGESTING!
    lexical_suggestion_generator = LexicalSuggestions(holder.light_verbs,
                                                      holder.light_verb_replacers,
                                                      holder.abstract_nouns,
                                                      holder.linguistic_analysis,
                                                      holder.law_abbreviations)

    lexical_suggestion_generator.start(changes_per_sentence)


    response['suggested_changes:'] = json.dumps(changes_per_sentence, ensure_ascii=False)


    # composer = ClusterComposer(response["content"])
    # response["Summary"] = str(composer.start(draw_clustering_optimization=False))
    return response


def store_data_in_data_holder_object(original_content) -> DataHolder:
    # DataHolder() object to store data for later re-use
    holder = DataHolder()

    # Splitting text to sentences, predict problematic sentences with fine-tined huBERT:
    preprocessor = PreprocessContent()
    problematic_sentences, sentences = preprocessor.start(original_content)

    init_data_holder(holder=holder)
    holder.splitted_sentences_from_text = sentences
    holder.problematic_sentences_from_splitted = problematic_sentences

    linguistic_analyzer = Analyzer() # holder.linguistic_analysis : dict[str, dict[str, List[str]]]
    holder.linguistic_analysis = linguistic_analyzer.start(splitted_sentences=holder.splitted_sentences_from_text)

    return holder


def init_data_holder(holder: DataHolder) -> None:
    df = pd.read_excel(str(files("resources") / "light_verb_constructions.xlsx"))
    to_replace = df["to_replace"].values.tolist()
    replacers = df["replacer"].values.tolist()
    holder.light_verbs = to_replace
    holder.light_verb_replacers = replacers
    with open(str(files("resources") / "abstracts.txt"), 'r', encoding='utf8') as abs:
        abstract_nouns = abs.readlines()
    holder.abstract_nouns = abstract_nouns
    with open(str(files("resources") / "law_abbreviations.txt"), 'r', encoding='utf8') as abbrev:
        abbreviations = abbrev.readlines()
        abbreviations = [abbr.replace('\n', '') for abbr in abbreviations]
    holder.law_abbreviations = abbreviations


def init_dict_changes(sentences: List[str], problematic_sentences: List[str]) -> Dict:
    """

    :param sentences: List of
    :param problematic_sentences:
    :return:
    """
    result_dict = {}
    for s in sentences:
        if s not in result_dict:
            result_dict[s] = ""
            if s not in problematic_sentences:
                result_dict[s] += "OK!"
            else:
                result_dict[s] = {'light_verbs': [],
                                  'too_abstract:': False,
                                  'verbless': False,
                                  'abbreviations': [],
                                  'too_long': False,
                                  'references': [],
                                  'too_much_clauses': False,
                                  'multiple_negations': False,
                                  'archaic': False}
    return result_dict
