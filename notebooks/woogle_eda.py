# %%
# manually set the correct types
import pandas as pd
import matplotlib.pyplot as plt

# where the data is stored
DATA_ROOT = './data/csv/'
# %%
# READ DATA
bodytext_dtypes = {'foi_documentId': str, 'foi_bodyTextOCR': str, 'foi_bodyText': str}
dossier_dtypes = {'dc_identifier': str, 'dc_publisher_name': str}
document_dtypes = {'foi_documentId': str, 'foi_dossierId': str, 'dc_title_doc': str, 'dc_source': str,
                   'foi_fileName': str, 'foi_nrPages': float}

print('Reading bodytext...')
bodytext_df = pd.read_csv(DATA_ROOT + 'bodytext.csv', dtype=bodytext_dtypes, usecols=bodytext_dtypes.keys())
print('Reading dossiers...')
dossier_df = pd.read_csv(DATA_ROOT + 'dossiers.csv', dtype=dossier_dtypes, usecols=dossier_dtypes.keys())
print('Reading documents...')
document_df = pd.read_csv(DATA_ROOT + 'documents.csv', dtype=document_dtypes, usecols=document_dtypes.keys())
# %%
# Total number of dossiers
number_of_dossiers = document_df['foi_dossierId'].unique().size
print('Total number of dossiers:', number_of_dossiers)
# %%
# Full histogram of the number of documents in the dossiers
document_df.groupby('foi_dossierId').size().hist()
plt.show()
# %%
# partial histogram of the number of documents in the dossiers
document_per_dossier = document_df.groupby('foi_dossierId').size()
document_per_dossier[document_per_dossier < 15].hist(bins=15)
plt.show()

# %%
# number of documents
number_of_documents = document_df['foi_documentId'].unique().size
print('Total number of documents:', number_of_documents)
# %%
# Full histogram of the number of pages in the documents
# WHERE IT WENT WRONG: foi_nrPages is not accurate, as we don't have pages for all documents
document_df['foi_nrPages'].hist()
plt.title('Number of pages in documents')
plt.xlabel('Number of pages')
plt.ylabel('Number of documents')
plt.show()
# %%
# Partial histogram of the number of pages in the documents
document_df[document_df['foi_nrPages'] < 25].hist(bins=25)
plt.title('Number of pages in documents')
plt.xlabel('Number of pages')
plt.ylabel('Number of documents')
plt.show()
# %%
# Number of documents with text
number_of_documents_with_text = document_df[document_df['foi_documentId'].isin(bodytext_df['foi_documentId'])]
print('Number of documents with text:', len(number_of_documents_with_text))
# document_df['foi_documentId'].unique().isin(document_bodytext_df['foi_documentId'].unique())
# %%
# number of pages per document, for documents we have text for
number_of_pages_per_doc = bodytext_df.groupby('foi_documentId').size()
number_of_pages_per_doc[number_of_pages_per_doc < 25].hist(bins=25)
plt.title('Number of pages in documents')
plt.xlabel('Number of pages')
plt.ylabel('Number of documents')
plt.show()
