from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import string
import gensim.downloader as api
# Function to convert a sentence (list of words) into a matrix representing the words in the embedding space
def embed_sentence(word2vec, sentence):
    embedded_sentence = []
    for c in '?!#*.-':
        sentence = sentence.replace(c," ")
    for word in sentence.split():
        if word.lower() in word2vec:
            embedded_sentence.append(word2vec[word.lower()])
    return np.array(embedded_sentence)
# Function that converts a list of sentences into a list of matrices
def embedding(word2vec, sentences):
    embed = []
    for sentence in sentences:
        embedded_sentence = embed_sentence(word2vec, sentence)
        embed.append(embedded_sentence)
    return embed
def pre_autoencoder(df):
  n_words = 25
  word2vec = api.load('glove-wiki-gigaword-50')
  X_train = df['review']
  # Embed the training and test sentences
  X_train_embed = embedding(word2vec, X_train)
  X_train_pad = pad_sequences(X_train_embed, dtype='float32', padding='post', maxlen=n_words)
  return X_train_pad
