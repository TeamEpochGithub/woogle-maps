import pandas as pd
from langdetect import detect
from nltk.tokenize import sent_tokenize
import spacy

# %%

dossier = pd.read_csv("./data/selected/formatted_documents.csv", usecols=['full_text_adj'])
# %%
# remove whitespace
doc_text = dossier.iloc[0]['full_text_adj']
# doc_text = ' '.join(doc_text.replace('\n', ' ').split())
# %%
sentences = sent_tokenize(doc_text)
# %%
for sentence in sentences:
    try:
        print(detect(sentence), sentence)
    except:
        pass

# %%
from spacy.pipeline.senter import DEFAULT_SENTER_MODEL
def filter_dutch_text(text):
    nlp = spacy.load("nl_core_news_sm")
    config = {"model": DEFAULT_SENTER_MODEL, }
    nlp.add_pipe("senter", config=config)
    doc = nlp(text)

    dutch_sentences = []
    for sent in doc.sents:
        if sent.lang_ == "nl":
            dutch_sentences.append(sent.text.strip())

    filtered_text = '\n'.join(dutch_sentences)
    return filtered_text

# %%
filtered_text = filter_dutch_text(doc_text)