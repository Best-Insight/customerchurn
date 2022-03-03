import numpy as np
import string
import gensim.downloader as api
import tensorflow as tf


# Function to convert a sentence (list of words) into a matrix representing the words in the embedding space
def embed_sentence(word2vec, sentence):
    embedded_sentence = []
    for c in string.punctuation:
        sentence = sentence.replace(c, "")

    for word in sentence.split():
        if word in word2vec:
            embedded_sentence.append(word2vec[word])

    return np.array(embedded_sentence)


# Function that converts a list of sentences into a list of matrices
def embedding(word2vec, sentences):
    embed = []
    for sentence in sentences:
        embedded_sentence = embed_sentence(word2vec, sentence)
        embed.append(embedded_sentence)
    return embed


def auto_encoder_preprocessing(sentences, n_words=20):
    word2vec = api.load('glove-wiki-gigaword-50')
    X_train_embed = embedding(word2vec, sentences)
    X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_embed,
                                dtype='float32',
                                padding='post',
                                maxlen=n_words)
    return X_train_pad
