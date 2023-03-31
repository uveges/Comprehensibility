import os

from importlib_resources import files

import config
from preprocessors.train_data_generation import Generator
from machine_learning.cross_validate_traditional import CrossValidation


def main():
    # Creating folder structure
    txt_folder = str(files("resources") / "txts")
    sentence_segmented_folder = str(files("resources") / "segmented")
    corpora_folder = str(files("resources") / "excel_corpora")
    initialize_folders([txt_folder, sentence_segmented_folder, corpora_folder])

    # Phase 1: generation training data from .doc, .docx files to classical ML algorithms (NB, SVM, LR etc.)
    # Resulting: 1-1 .xlsx file contains the rephrased, to-be-rephrased sentences with its label in a different column

    start_data_generation(txt_folder=txt_folder,
                          sentence_segmented_folder=sentence_segmented_folder,
                          corpora_folder=corpora_folder,
                          dump_subcorpora_per_file=False)

    # Phase 2: carry out ML training with different algoritmhs
    find_optimal_classic_model(os.path.join(corpora_folder, "full_dataset.xlsx"))


def find_optimal_classic_model(dataframe_path: str):
    validator = CrossValidation(dataframe_path, preprocess=True)
    validator.start()


def start_data_generation(txt_folder: str,
                          sentence_segmented_folder: str,
                          corpora_folder: str,
                          dump_subcorpora_per_file=False):
    generator = Generator(mode="splitter",
                          name_original="original.xlsx",
                          name_rephrased="rephrased.xlsx",
                          name_full="full_dataset.xlsx",
                          corpora_folder=corpora_folder,
                          txt_folder=txt_folder,
                          sentence_segmented_folder=sentence_segmented_folder
                          )

    csak_eredeti = []
    csak_kozertheto = []

    generator.docxtotxt(config.WORD_FILES_FOLDER)
    generator.segment_txts_to_sentences(debug=False)

    FILENAMES = generator.get_segmented_filenames()  # összegyűjtjük a mondat-szegmentált fájlok neveit
    WORD_DOCUMENT_TRIPLETS = generator.gettriplets(FILENAMES)

    dump = []

    for word in WORD_DOCUMENT_TRIPLETS:
        csak_eredeti.append(word.get_subcorpora()[0])  # elso lista: CSAK EREDETI, második: CSAK KÖZÉRTHETŐ
        csak_kozertheto.append(word.get_subcorpora()[1])
        if dump_subcorpora_per_file:
            dump.append("Original:")
            dump.append(word.get_subcorpora()[0])
            dump.append("Rephrased:")
            dump.append(word.get_subcorpora()[1])
            dump.append("___________")

    if dump_subcorpora_per_file:
        with open(config.MAIN_FOLDER_FOR_PROJECT + "/per_file_sentences.txt", "w", encoding='utf8') as output:
            for item in dump:
                output.write(str(item))
        print(f'Written dump to: {config.MAIN_FOLDER_FOR_PROJECT + "/per_file_sentences.txt"}!')

    generator.covert_to_excels(csak_eredeti, "original.xlsx")
    generator.covert_to_excels(csak_kozertheto, "rephrased.xlsx")

    generator.create_training_dataset()


def initialize_folders(folders: list) -> None:
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


if __name__ == '__main__':
    main()
