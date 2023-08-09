import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

text = """ """ # The text you want to summarize should go here
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

stopwords = set(STOP_WORDS)
punctuation = punctuation + '\n'

word_frequencies = {}
for word in doc:
    if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
        word_frequencies[word.text] = word_frequencies.get(word.text, 0) + 1

max_frequency = max(word_frequencies.values())
word_frequencies = {word: freq / max_frequency for word, freq in word_frequencies.items()}

sentence_tokens = list(doc.sents)

sentence_scores = {}
for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies:
            sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word.text.lower()]

select_length = int(len(sentence_tokens) * 0.2)
summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

final_summary = ' '.join([word.text for word in summary])

