from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd


def remove_punctuations(text):
    punctuations = string.punctuation
    for punctuation in punctuations:
        text = text.replace(punctuation, '')
    return text

def lowercase(text):
    text = text.lower()
    return text

def remove_num(text):
    text = ''.join(word for word in text if not word.isdigit())
    return text

# def lemmatize(text):
#     nlp = spacy.load("en_core_web_sm")
#     text = ' '.join(token.lemma_ for token in nlp(text))
#     return text

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in word_tokens]
    return ' '.join(lemmatized)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    word_tokens = [word for word in word_tokens if not word in stop_words]
    return ' '.join(word_tokens)

def text_prepro(text):
    text = lowercase(text)
    text = remove_punctuations(text)
    text = remove_num(text)
    text = lemmatize(text)
    text = remove_stopwords(text)
    return text

def series_prepro(series):
    return pd.DataFrame(series.apply(text_prepro))
