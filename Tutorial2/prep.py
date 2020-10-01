"""
This file will prepare our raw data into a more contained version of itself
"""

import os
import re
import sys
import spacy
import numpy as np
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS

df = pd.read_csv(
    '../../Data/training.1600000.processed.noemoticon.csv', encoding='latin1', header=None)

# Explore data

df = df[[5, 0]]
df.columns = ['twitts', 'sentiment']

df['word_cout'] = df['twitts'].apply(lambda x: len(str(x).split()))


def get_avrg_word_len(x):
    words = x.split()
    word_len = 0
    for word in words:
        word_len = word_len + len(word)
    return word_len/len(words)


df['word_avrg_len'] = df['twitts'].apply(lambda x: get_avrg_word_len(x))


# stop words len
df['stop_words_len'] = df['twitts'].apply(
    lambda x: len([words for words in x.split() if words in STOP_WORDS]))


# look for characters
def sample_char_fun(x):
    [word for word in x.split() if word.startswith('@')]

# locate words specifically:
    df.loc['row']['column']

#Preprocessing and cleaning


# Lower case
df['twitts'] = df['twitts'].apply(lambda x: x.lower())

# Contractions
# explanation
contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how does",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    " u ": " you ",
    " ur ": " your ",
    " n ": " and ",
    "won't": "would not",
    "dis": "this",
    "brng": "bring"
}


def cont(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x


sample = "i'd be happy"

print(cont(sample))


# remove emails

# remove most frequently used words..?
text = ' '.join(df['twitts'])
text = text.split()
freq_comm = pd.Series(text).value_counts()
f20 = freq_comm[:20]

df['twitts'].apply(lambda x: " ".join(
    [words for words in x.split() if words not in f20]))

# remove least frequently used word
rare20 = freq_comm[-20:]
rare = freq_comm[freq_comm.values == 1]
df['twitts'].apply(lambda x: ' '.join(
    [words for words in x.split() if words not in rare]))
