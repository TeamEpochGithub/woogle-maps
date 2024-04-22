import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import sklearn
from nltk.tokenize import sent_tokenize
from matplotlib import pyplot as plt

# %%
df = document_df = pd.read_csv("./data/selected/formatted_documents.csv", usecols=['full_text_adj'])
# %%
# read in a file wordlist.txt
with open('./data/wordlist.txt', 'r') as file:
    wordlist = file.read().splitlines()
# %%
# remove whitespace multiple whitespaces
doc_text: str = document_df.iloc[0]['full_text_adj']

def remove_shitty_sentences(doc_text: str) -> str:
    print('Original text:', doc_text)
    doc_text = re.sub(r'\n+', '\n', doc_text)
    doc_text = re.sub(r' +', ' ', doc_text)
    # filter out words not in the wordlist
    doc_text = re.sub(r'\n', '. ', doc_text)
    sentences = sent_tokenize(doc_text, language='dutch')
    sentences_good = [' '.join([word for word in sentence.split() if word in wordlist]) for sentence in sentences]
    sentences_good = [sentence for sentence in sentences_good if len(sentence.split()) > 2]


def remove_shitty_sentences_df(df: pd.DataFrame) -> pd.DataFrame:
    # replace multiple whitespace multiple whitespaces
    # replace newlines with dots for the sentence tokenizer
    print('Replacing newlines and multiple whitespaces')
    df['full_text_adj'] = df['full_text_adj'].fillna('')
    df['filtered_text'] = df['full_text_adj'].str.replace(r'\n+', '\n')
    df['filtered_text'] = df['filtered_text'].str.replace(r' +', ' ')
    print('Tokenizing sentences')
    df['filtered_text'] = df['filtered_text'].str.replace(r'\n', '. ')
    print('Tokenizing sentences starting')
    # filtered text a list of sentences
    df['filtered_text'] = df['filtered_text'].apply(lambda x: sent_tokenize(x, language='dutch'))
    df['filtered_text'].explode()
    print('Filtering out words not in the wordlist')
    df['filtered_text'] = df['filtered_text'].apply(
        lambda x: [' '.join([word for word in sentence.split() if word in wordlist]) for sentence in x])
    df['filtered_text'] = df['filtered_text'].apply(lambda x: [sentence for sentence in x if len(sentence.split()) > 2])
    return df
# %%

print('Replacing newlines and multiple whitespaces')
df['full_text_adj'] = df['full_text_adj'].fillna('')
df['filtered_text'] = df['full_text_adj'].str.replace(r'\n+', '\n', regex=True)
df['filtered_text'] = df['filtered_text'].str.replace(r' +', ' ', regex=True)
# %%
print('Tokenizing sentences')
df['filtered_text'] = df['filtered_text'].str.replace(r'\n', '. ')
# filtered text is a list of sentences
df['filtered_text'] = df['filtered_text'].apply(lambda x: sent_tokenize(x, language='dutch'))
# %%
# Step 1: Explode text into sentences
df_exploded = df.explode('filtered_text')['filtered_text'].str.split().explode()
# explode the sentences into words
# df_exploded['filtered_text'] = df_exploded['filtered_text']
# %%
# Step 2: Filter words based on 'wordlist'
df_filtered = df_exploded[df_exploded.isin(wordlist)]
# %%

# Step 3: Aggregate words back into sentences
df_aggregated = df_filtered.groupby(level=0, sort=False).apply(' '.join).reset_index()
# %%
df_aggregated['full_text_adj'] = df['full_text_adj']
# %%
# df_aggregated.rename(columns={'filtered_text': 'filtered_text_aggregated'}, inplace=True)
# df['filtered_text'].explode()
# print('Filtering out words not in the wordlist')
# df['filtered_text'] = df['filtered_text'].apply(
#     lambda x: [' '.join([word for word in sentence.split() if word in wordlist]) for sentence in x])
# df['filtered_text'] = df['filtered_text'].apply(lambda x: [sentence for sentence in x if len(sentence.split()) > 2])


# %%
df = remove_shitty_sentences_df(document_df)
# %%

df_aggregated[['full_text_adj', 'filtered_text']].style.set_table_styles([
    {
        'selector': '',
        'props': [('font-family', 'Arial, sans-serif'), ('border-collapse', 'collapse'), ('width', '100%')]
    }, {
        'selector': 'td, th',
        'props': [('border', '1px solid #dddddd'), ('text-align', 'left'), ('padding', '8px')]
    }, {
        'selector': 'tr:nth-child(even)',
        'props': [('background-color', '#f2f2f2')]
    }, {
        'selector': 'th',
        'props': [('background-color', '#4CAF50'), ('color', 'white')]
    }
]).to_html('./data/selected/formatted_documents.html')

# %%
# remove sentences if they are one or two words
# doc_text = re.sub(r'\.\b\w{1,2}\.', '', doc_text)
# doc_text = ' '.join(doc_text.replace('\n', '. ').split())
# %%
# sentences = re.split('\\. |\\? |! ', doc_text)
# %%
# plot the length of the sentences
plt.hist([len(sentence.split()) for sentence in sentences_good], bins=40)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()
plt.hist([len(sentence.split()) for sentence in sentences], bins=40)
plt.ylim(0, 20)
plt.xlim(0, 20)
plt.show()

# doc_text = ' '.join([word for word in doc_text.split() if word in wordlist])
# %%
print('Preprocessed text', '\n'.join(sentences))
# sentences.append('Ik ging gisteren lunchen met een vriend, we dronken ook een beetje wijn, maar we wisten niet hoe we het evenement konden laten plaatsvinden.')
# %%
model = SentenceTransformer("all-MiniLM-L6-v2")

# %%
# Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)
# %%
# average the embeddings
average_embedding = np.mean(embeddings, axis=0)
# %%
# rank embeddings based on cosine similarity to the average embedding with numpy
threshold = 0.72
cos_sim = sklearn.metrics.pairwise.cosine_similarity(embeddings, average_embedding.reshape(1, -1))
for sentence, sim in zip(sentences, cos_sim):
    print("✅" if sim > threshold else "", "Sentence:", sentence)
    print("Cosine-Similarity:",
          sim)
    print("")
# noise = model.encode(['Ik ging gisteren lunchen met een vriend, we dronken ook een beetje wijn, maar we wisten niet hoe we het evenement konden laten plaatsvinden.'])
# %%
noise = model.encode(['Ik ging gisteren lunchen met een vriend.'])
noise_sim = sklearn.metrics.pairwise.cosine_similarity(noise, average_embedding.reshape(1, -1))
print("Noise:", noise_sim)
# %%
cos_sim = cos_sim.flatten()
# %%
good_sentences = [sentence for sentence, sim in zip(sentences, cos_sim) if sim > threshold]
