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
from textblob import TextBlob
from spacy.lang.en.stop_words import STOP_WORDS

# Read Data and examine it

df = pd.read_csv(
    '../../Data/training.1600000.processed.noemoticon.csv', encoding='latin1', header=None)

df = df[[5, 0]]

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


print(df['Word_count'])
print('---------------------')
print(df['Stop_word_count'])
print('---------------------')
print(df['Char_count'])
print('---------------------')
print(df['Punctuation_count'])
print('---------------------')
print(df['Mention_count'])
print('---------------------')
print(df['Hashtag_count'])
print('---------------------')
print(df['Avrg_word_len'])
print('---------------------')


# Find most frequent and rarest word ocrruences
