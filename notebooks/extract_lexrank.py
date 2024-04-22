import pandas as pd
from lexrank import LexRank
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import pickle

# %%

def remove_shitty_sentences_df(df: pd.DataFrame) -> pd.DataFrame:
    # replace multiple whitespace multiple whitespaces
    # replace newlines with dots for the sentence tokenizer
    print('Replacing newlines and multiple whitespaces')
    df['full_text_adj'] = df['full_text_adj'].fillna('')
    df['filtered_text'] = df['full_text_adj'].str.replace(r'\n+', '\n')
    df['filtered_text'] = df['filtered_text'].str.replace(r' +', ' ')
    print('Tokenizing sentences')
    df['filtered_text'] = df['filtered_text'].str.replace(r'\n', '. ')
    # filtered text a list of sentences
    df['filtered_text'] = df['filtered_text'].apply(lambda x: sent_tokenize(x, language='dutch'))
    return df


# %%
print('Reading bodytext')
bodytext_df = pd.read_csv("./data/selected/formatted_documents.csv")
# %%
print('Removing shitty sentences from bodytext')
bodytext_df = bodytext_df.rename({'foi_bodyText': 'full_text_adj'}, axis=1)
df = remove_shitty_sentences_df(bodytext_df)
# %%
print('Training lexrank model')
lxr = LexRank(df['filtered_text'].to_list(), stopwords=set(stopwords.words('dutch')))
# %%
pickle.dump(lxr, open('./data/selected/lexrank.pkl', 'wb'))
# %%
sentences = df.iloc[0]['filtered_text']

# get summary with classical LexRank algorithm
df['summary'] = df['full_text_adj'].apply(lxr.get_summary(sentences, summary_size=len(sentences) // 2, threshold=.1))
