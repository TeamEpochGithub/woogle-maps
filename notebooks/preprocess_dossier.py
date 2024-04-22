# %%
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch

SELECTED_DOSSIER_ID = 'nl.gm0867.2i.2023.11'

# %%
# The example urls do not work, at least for the great majority of the examples.
formatted_documents = pd.read_csv('data/processed/nl.gm0867.2i.2023.11/formatted_documents.csv')
documents = pd.read_csv('data/WoogleDumps/woo_documents.csv.gz', low_memory=False)
#final_dossier = pd.read_csv('data/final_dossier.csv')

# %%
df_final = formatted_documents[10:20] # Take arbitrary number of documents, depending each use case
df_final.drop(columns=['foi_dossierId','foi_fileName', 'foi_documentId','Unnamed: 0'], inplace=True)

# %%
# Retrieve urls from documents
# Now, update 'url' in df_final with 'dc_source' from the merge
for idx, row in df_final.iterrows():
    doc = documents[documents['dc_identifier'] == row['dc_identifier']]
    print(doc['dc_source'])
    if not row.empty:
        df_final.loc[idx, 'embedding'] = doc['dc_source'].iloc[0]

# %%
# Generate embeddings
tokenizer = RobertaTokenizer.from_pretrained("tm/bert-tokenizer")
model = RobertaModel.from_pretrained("tm/bert-embedder")

bodies = df_final['full_text'].tolist()
encoded_bodies = tokenizer(bodies, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**encoded_bodies)

# Extract the embeddings
embeddings = outputs.last_hidden_state
embedding_list = embeddings.numpy().tolist()
df_final['embedding'] = embedding_list #Throws caveat warning

# %%
# Export to csv
df_final.rename(columns={'dc_identifier': 'doc_id'})
df_final.to_csv('data/final/processed_dossier.csv')
