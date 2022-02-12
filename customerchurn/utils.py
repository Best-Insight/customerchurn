import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


def get_lang_tag(reviews):

    def get_lang_detector(nlp, name):
        return LanguageDetector()

    nlp = spacy.load("en_core_web_sm")
    Language.factory("language_detector01", func=get_lang_detector)
    nlp.add_pipe('language_detector01', last=True)

    return [nlp(str(row))._.language['language'] for row in reviews]


def stars_2_recoms(series, star=4):

    def get_recom(s):
        return 1 if s >= star else 0

    return [get_recom(s) for s in series]
