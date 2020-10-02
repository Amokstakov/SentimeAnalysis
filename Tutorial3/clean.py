"""
This Tutorial will be my solo attempt at classifying Twitter sentiment using all the knowledge I have gathered thus far.
This particular script will focus on cleaning the Twitter data.

Data source: Kazanova from Kaggle.

Knowledge acquired: Various Medium articles, KGPTalkie, Machine Learning Mastery

Method Used: Cleaning techniques will be my own.

Model Used: Word Embeddings.


Goal:
    Acquire certain training accuracy and avoid overfitting
"""

# Imports

# System imports
import os
import re
import sys

import spacy
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from prep import contractions
from textblob import TextBlob
from tensorflow.keras.models import Sequential
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Flatten, Dropout, Embedding, Activation, Conv1D, MaxPooling1D, GlobalMaxPooling1D


nlp = spacy.load('en_core_web_md')


# Read Data and examine it

# df = pd.read_csv(
# '../../Data/training.1600000.processed.noemoticon.csv', encoding='latin1', header=None)


df = pd.read_csv('../../Data/twitter-data-master/twitt30k.csv',
                 encoding='latin1', header=None)

print(df)
sys.exit()

df.columns = ['Tweets', 'Sentiment']

"""
Look at Meta Features such as:
    Word_count
    stop_word_count
    url_count
    avr_word_len
    char_len
    punctuation count
    hashtag_count -> Twitter Specific
    mention count -> Twitter Specific
"""

# Word Count
df['Word_count'] = df['Tweets'].apply(lambda x: len(str(x).split()))

# stop_word count
df['Stop_word_count'] = df['Tweets'].apply(lambda x: len(
    [word for word in x.split() if word in STOP_WORDS]))

# get character count
df['Char_count'] = df['Tweets'].apply(lambda x: len(str(x)))

# punctuation count
df['Punctuation_count'] = df['Tweets'].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation]))

# get @ mention count
df['Mention_count'] = df['Tweets'].apply(
    lambda x: len([t for t in x.split() if t.startswith('@')]))

# get hashtags
df['Hashtag_count'] = df['Tweets'].apply(
    lambda x: len([t for t in x.split() if t.startswith('#')]))

# get average word len


def get_avrg_word_len(x):
    words = x.split()
    word_len = 0
    for word in words:
        word_len = word_len + len(word)
    return word_len/len(words)


df['Avrg_word_len'] = df['Tweets'].apply(lambda x: get_avrg_word_len(x))


# Find most frequent and rarest word ocrruences
text = ' '.join(df['Tweets'])
text = text.split()
freq_ = pd.Series(text).value_counts()
Top_10 = freq_[:10]

Least_freq = freq_[freq_.values == 1]


# Clean the data

def contractions_replace(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    return x


def get_clean_data(x):
    # go through multiple steps of cleaning

    # turn everything into lower case
    x = x.lower()

    # fix all potential spelling mistakes
    x = TextBlob(x).correct()

    # remove all emails
    x = re.sub('([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', "", x)

    # remove all @ first
    x = re.sub(r'@([A-Za-z0-9_]+)', "", x)

    # remove all # first
    x = re.sub(r'#([A-Za-z0-9_]+)', "", x)

    # remove and strip all retweets (RT)
    x = re.sub(r'\brt:\b', '', x).strip()

    # remove all websites
    # TODO: Figure out how it works for all possible website protocols
    x = re.sub(
        r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x)

    # clean and replace with contractions
    x = contractions_replace(x)

    # remove all numerical values
    x = re.sub(r'[0-9]+', "", x)

    # remove all special characters
    x = re.sub(r'[^\w ]+', ' ', x)

    # split aka tokenize our tweets
    x = x.split()

    # We are removed all the workds that are in our top 10
    x = [words for words in x if words not in Top_10]

    # We are rempoving all the words that are not in our rare list
    x = [words for words in x if words not in Least_freq]

    # remove all the words in our STOP_WORDS
    x = [words for words in x if words not in STOP_WORDS]

    return " ".join(x)


# Model build

df['Tweets'] = df['Tweets'].apply(lambda x: get_clean_data(x))

text = df['Tweets'].tolist()

y = df['Sentiment']

# Create Tokenizer instance
token = Tokenizer()
# Fit The Texts on our tokenizr
token.fit_on_texts(text)

vocab_size = len(token.word_index) + 1

# encode the text
encoded_text = token.texts_to_sequences(text)

# Pad the sequences
max_len = max([len(s, split()) for s in text])

X = pad_sequences(encoded_text, maxlen=max_len, padding='post')

# Create the embedding vectors using GloVe
glove_vectors = dict()
file = open('../../Data/glove.twitter.27B.200d.txt', encoding='utf-8')

for lin in file:
    value = line.split()
    word = value[0]
    vector = np.asarray(value[1:])
    glove_vectors[word] = vector
file.close()


# our task is to get the global vectors for our words
# create empty matrix with the proper size
word_vector_matrix = np.zeros((vocab_size, 200))

for word, index in token.word_index.items():
    vector = glove_vectors.get(word)
    # check if the word is not present in GloVe
    if vector is not None:
        word_vector_matrix[index] = vector
    else:
        print(word)


x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y)

vec_size = 200

model = tf.keras.Sequential()
model.add(Embedding(vocab_size, vec_size, input_length=max_len,
                    weights=[word_vector_matrix], trainable=False))
model.add(Conv1D(64, 8, activation="relu"))
model.add(MaxPooling1D(2))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test))

