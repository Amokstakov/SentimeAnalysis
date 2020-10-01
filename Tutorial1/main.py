"""
This tutorial comes from: 
    https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/
"""
import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS


nlp = spacy.load('en_core_web_sm')

nlp.add_pipe(nlp.create_pipe('sentencizer'))

text = """
When learning data science, you shouldn’t get discouraged.
Challenges and setbacks aren’t failures, they’re just part of the journey.
New York city, is a plce to go in September. I also work in Google
"""

doc = nlp(text)

for token in doc:
    print(token, token.pos_)

for sent in doc.sents:
    print(sent)

# removing stop words
filtered_words = []

for token in doc:
    if token.is_stop == False:
        filtered_words.append(token)
print(filtered_words)

lem = nlp("run running runs runnner")
for token in lem:
    print(token.text, token.lemma_)

# Entities
for token in doc.ents:
    print(token, token.label_, token.label)
