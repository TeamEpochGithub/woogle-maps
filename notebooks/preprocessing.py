# %%
from datetime import datetime
from pathlib import Path

import pandas as pd

# where the dataset is stored
DATA_ROOT = './data/WoogleDumps/'
SELECTED_DOSSIER_ID = 'nl.gm0867.2i.2023.11'


# %%

def read_all() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Read all data from the csv files located in DATA_ROOT
    :return:
    """
    bodytext_dtypes = {'foi_documentId': str, 'foi_bodyTextOCR': str}
    dossier_dtypes = {'dc_identifier': str, 'dc_publisher_name': str}
    document_dtypes = {'dc_identifier': str, 'foi_dossierId': str, 'dc_title': str, 'dc_source': str,
                       'foi_fileName': str}

    print('Reading bodytext...')
    bodytext_df = pd.read_csv(DATA_ROOT + 'woo_bodytext.csv.gz', dtype=bodytext_dtypes, usecols=bodytext_dtypes.keys())
    print('Reading dossiers...')
    dossier_df = pd.read_csv(DATA_ROOT + 'woo_dossiers.csv.gz', dtype=dossier_dtypes, usecols=dossier_dtypes.keys())
    print('Reading documents...')
    document_df = pd.read_csv(DATA_ROOT + 'woo_documents.csv.gz', dtype=document_dtypes, usecols=document_dtypes.keys())

    return bodytext_df, dossier_df, document_df


# %%
def join_all(bodytext_df: pd.DataFrame, dossier_df: pd.DataFrame, document_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join all dataframes into one
    :param bodytext_df: dataframe with text, each page is a row
    :param dossier_df: dataframe with dossier metadata
    :param document_df: dataframe with document metadata
    :return:
    """
    # Filter: only keep dossiers and documents with a certain identifier
    dossier_df = dossier_df[dossier_df['dc_identifier'] == SELECTED_DOSSIER_ID].copy(deep=False)
    document_df = document_df[document_df['foi_dossierId'] == SELECTED_DOSSIER_ID].copy(deep=False)

    document_df['dc_source'] = document_df['dc_source'].fillna('')
    document_df['dc_title'] = document_df['dc_title'].fillna(document_df['foi_fileName']).fillna('')

    # set the publisher name from the dossier
    publisher: str = dossier_df.iloc[0]['dc_publisher_name']
    document_df['dc_publisher_name'] = publisher

    # filter, join pages
    bodytext_df['foi_bodyTextOCR'] = bodytext_df['foi_bodyTextOCR'].fillna('')
    bodytext_df = bodytext_df[bodytext_df['foi_documentId'].isin(document_df['dc_identifier'])]
    bodytext_df = bodytext_df.groupby('foi_documentId').agg(full_text=('foi_bodyTextOCR', '\n'.join)).reset_index()

    # join document metadata with its text
    return document_df.merge(bodytext_df, left_on='dc_identifier', right_on='foi_documentId', how='left')


# %%
# bodytext_df, dossier_df, document_df = read_all()
# df = join_all(bodytext_df, dossier_df, document_df)
# OR
df = join_all(*read_all())  # READ EVERYTHING, CHOOSE ANY DOSSIER
# %%
# insert id column
df['id'] = range(0, len(df))
# set date to now, we don't know the original date yet
df['date'] = datetime.now()
# %%
rename_cols = {
    'dc_title': 'title',
    'dc_source': 'url',
    'dc_publisher_name': 'publication'
}
df = df.rename(columns=rename_cols)
# %%
Path(f'./data/processed/{SELECTED_DOSSIER_ID}').mkdir(parents=True, exist_ok=True)
df.to_csv(f'./data/processed/{SELECTED_DOSSIER_ID}/formatted_documents.csv', sep=',')
