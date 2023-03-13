import nltk
from investigations.validate_magyarlanc import ValidateMagyarlanc
from investigations.validate_sentence_splitter import ValidateSentenceSplitter
from investigations.validate_spacy import ValiateSpacyModel

def main():

    base_dir = "/home/istvanu/Documents/sentence_segmentation"

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    validator1 = ValidateMagyarlanc(base_dir)
    validator1.validate_magyarlanc()

    validator2 = ValidateSentenceSplitter(base_dir)
    validator2.validate()

    validator3 = ValiateSpacyModel(base_dir)
    validator3.validate()


if __name__ == '__main__':
    main()
