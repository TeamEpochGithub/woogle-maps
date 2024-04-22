# %% [markdown]
# # Woogle Database Documentation
# 
# This notebook contains a detailed documentation of the dumps of the woogle datasets, as well as a short exploration into the characteristics of the data, and some interesting usecases.
# 
# The [Woogle search engine](https://woogle.wooverheid.nl) is part of the [Wooverheid](https://wooverheid.nl) Project, aimed at providing access to Woo documents from the government in an online computer-friendly format.
# Currently, Woogle contains the Woo dossiers of a little more over a million pages, and is growing daily.
# Woogle continas documents released under the WOO act from all 17 information categories, and provides quickly and easy access to the information in these documents through the search engine.
# 
# One of the main goals of Woogle is the show the possibilities when government data is open and easily accessible, both to researchers and to interested civilians and institutions. By providing the data from the requests in a macine-readable format, this offers the opportunity to do many interesting research into language usage, text redaction, etc.
# 
# **Authors**: Maik Larooi, Maarten Marx, Jaap Kamps, Ruben van Heusden

import matplotlib.pyplot as plt

# %matplotlib inline

# %% [markdown]
# ## The dataframes
# 
# The database dumps contain the information in 3 different dataframes, namely the `woo_bodytext`, `woo_dossiers` and `woo_documents` dataframes. Most importantly, the `woo_bodytext` contains the text of all pages from the dataset, along with an analysis of the amount of redaction applied in documents. The `file_dataframe` contains the data about the complete dossiers received, and the `documents_dataframe` contains the data about the specific documents from the dossiers.  At the end of the notebook we present a detailed description of the 3 data frames and all the columns they contain.

# %% [markdown]
# <a id="loading" />

# %% [markdown]
# ## Loading
# 
# First, we're going to load the dataframe into `pandas`, and make sure all the types of the columns are correct with what they should contain. These should already be pretty good, but we will specifically convert the columns with dates to the `datetime` format, this works much better if we want to do something with dates.

# %%
# manually set the correct types
import pandas as pd
import matplotlib.pyplot as plt
bodytext_dtypes = {'dc_publisher_name': str, 'dc_publisher': str, 'foi_documentId': str, 'foi_pageNumber': int,
                   'foi_bodyText': str, 'foi_bodyTextOCR': str, 'foi_hasOCR': bool, 'foi_redacted': float,
                   'foi_contourArea': float, 'foi_textArea': float, 'foi_charArea': float,
                   'foi_percentageTextAreaRedacted': float, 'foi_percentageCharAreaRedacted': float,
                   'foi_nrWords': int, 'foi_nrChars': int, 'foi_nrWordsOCR ': int, 'foi_nrCharsOCR': int}

dossier_dtypes = {'dc_identifier': str, 'dc_title': str, 'dc_description': str, 'dc_type': str,
                  'foi_type_description': str, 'dc_publisher_name': str, 'dc_publisher': str,
                  'dc_source': str, 'foi_valuation': str, 'foi_requestText': str,
                  'foi_decisionText': str, 'foi_isAdjourned': str, 'foi_requester': str}

document_dtypes = {'dc_identifier': str, 'foi_dossierId': str, 'dc_title': str, 'foi_fileName': str,
                   'dc_format': str, 'dc_source': str, 'dc_type': str, 'foi_nrPages': float}

# %%
data_root = './data/WoogleDumps/'
n_rows_to_load = 1e5

# %%
nan_count = 0
total_len = 0
i = 0
for bodytext_df in pd.read_csv(data_root + 'woo_bodytext.csv.gz', dtype=bodytext_dtypes,  chunksize=n_rows_to_load):
    i += 1
    print('Loaded Chunk: ', i)
    nan_count += bodytext_df['foi_bodyText'].isna().sum()
    total_len += len(bodytext_df)
    print('Nan count: ', nan_count)
    print('Total length: ', total_len)

# %%
bodytext_iter = pd.read_csv(data_root + 'woo_bodytext.csv.gz', dtype=bodytext_dtypes,  chunksize=n_rows_to_load)
bodytext_df = [chunk[chunk['foi_bodyText'].notna()] for chunk in bodytext_iter]

# %%
bodytext_df = pd.read_csv(data_root + 'woo_bodytext.csv.gz', dtype=bodytext_dtypes, nrows=n_rows_to_load)
# %%
# ------------------------ Filter ------------------------
print('OCR missing', bodytext_df['foi_bodyTextOCR'].isna().sum() / len(bodytext_df))
print('Aligned text missing', bodytext_df['foi_bodyText'].isna().sum() / len(bodytext_df))
bodytext_df = bodytext_df.dropna(subset=['foi_bodyText'])
# %%
bodytext_df = bodytext_df.drop_duplicates(subset=['foi_bodyText'])
# %%
bodytext_df['foi_bodyText'] = bodytext_df['foi_bodyText'].fillna('')
# get the length of the texts
bodytext_df['foi_bodyText'] = bodytext_df['foi_bodyText'].str.replace('\n', '')
bodytext_df['foi_bodyText'] = bodytext_df['foi_bodyText'].str.split(' ')
# %%
# remove stopwords
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('dutch')) | set(stopwords.words('english')) | set(['.', ',', '(', ')', '!', '?', ':', ';', '``', "''", '’', '“', '”', '–','\n'])
# %%
bodytext_df['foi_bodyText'] = bodytext_df['foi_bodyText'].apply(lambda x: [word for word in x if word.lower() not in stop_words])


# %%
from collections import Counter
# Flatten the lists of words
bodytext_df['foi_bodyText'] = bodytext_df['foi_bodyText'].fillna('')
all_words = [word for sublist in bodytext_df['foi_bodyText'] for word in sublist]

# Count the frequency of each word
word_counts = Counter(all_words)
# %%
# Convert the Counter to a DataFrame
word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index', columns=['Frequency'])

# Sort the DataFrame by frequency
word_counts_df = word_counts_df.sort_values(by='Frequency', ascending=False)

print(word_counts_df)
# %% get the word frequency
word_frequency = bodytext_df['foi_bodyText'].apply(pd.Series).stack().value_counts()
word_count = bodytext_df['foi_bodyText'].apply()

# %%
bodytext_df['length'] = bodytext_df['foi_bodyText'].apply(len)
std = bodytext_df['length'].std()
mean = bodytext_df['length'].mean()
# %%
bodytext_df = bodytext_df[bodytext_df['length'] > 10]
bodytext_df = bodytext_df[bodytext_df['length'] < mean + 2 * std]
bodytext_df = bodytext_df[bodytext_df['length'] > mean - 2 * std]
# %%
print('Max: ', bodytext_df['length'].max())
print('Min: ', bodytext_df['length'].min())
# %%
bodytext_df['length'].plot(kind='hist', bins=200, x='Number of words', y='Number of pages')
plt.xlim(0, 3000)
plt.xlabel('Number of words')
plt.ylabel('Number of pages')
plt.title('Histogram of the number of words in the bodytext')
plt.show()
# %%
# bodytext_df[bodytext_df['length'] == 2]['foi_bodyText'].values
# %%
bodytext_df[bodytext_df['foi_bodyText'].str.fullmatch('\\b.*\\s\\n')]
# %%
bodytext_df = pd.read_csv(data_root + 'woo_bodytext.csv.gz', dtype=bodytext_dtypes)
dossier_df = pd.read_csv(data_root + 'woo_dossiers.csv.gz',
                          parse_dates=['foi_publishedDate', 'dc_date_year', 'foi_requestDate',
                                              'foi_decisionDate',
                                              'foi_retrievedDate'], dtype=dossier_dtypes)
document_df = pd.read_csv(data_root + 'woo_documents.csv.gz', dtype=document_dtypes)
# manually changes some columns, has to be done seperately due to nan values
# bodytext_df['foi_redacted'] = bodytext_df['foi_redacted'].astype(bool)
# %%
document_df['foi_nrPages'].isna().sum()
# %%
document_df = document_df.dropna(subset=['foi_nrPages'])
# %%
document_df['foi_nrPages'] = document_df['foi_nrPages'].astype(int)
# %%
std = document_df['foi_nrPages'].std()
mean = document_df['foi_nrPages'].mean()
print('Max: ', document_df['foi_nrPages'].max())
print('Min: ', document_df['foi_nrPages'].min())
# %%
other_df = document_df[document_df['foi_nrPages'] < mean + 4 * std]
print('Max: ', other_df['foi_nrPages'].max())
print('Min: ', other_df['foi_nrPages'].min())
# %%
len(document_df[document_df['foi_nrPages'] > 3]) / len(document_df)
# %%
document_df[document_df['foi_nrPages'] < 25]['foi_nrPages'].plot(kind='hist', bins=25, x='Number of pages', y='Number of documents')
plt.xlim(1, 15)
plt.xlabel('Number of pages')
plt.title('Histogram of the number of pages in the documents')
plt.savefig('plots/number_of_pages_documents.png', bbox_inches='tight')
plt.show()
# %%
# get the longest request
dossier_df[dossier_df['foi_requestText'].notna()]['foi_requestText'].apply(len).sort_values(ascending=False)

# %% [markdown]
# To get a first glance on what is contained in the datatframes, we will print the head of the dataframes and show how many columns are in each of the dataframes.
# %%
print(
    "The bodytext dataframe with the information from all pages contains %d rows and %d columns" % bodytext_df.shape)
bodytext_df.head(5)
# %%
print(
    "The document dataframe with the information from all pages contains %d rows and %d columns" % document_df.shape)
document_df.head(5)
# %%
print(
    "The dossier dataframe with the information from all pages contains %d rows and %d columns" % dossier_df.shape)
dossier_df.head(5)
# %%
dossier_df[dossier_df.dc_type == '2e-b']

# %%
# small funtion to print nice decimal numbers
add_dot = lambda x: format(x, ',')

print("In total there are  %s pages in the current woogle dump" % add_dot(bodytext_df.shape[0]))
print("In total there are  %s dossiers in the current woogle dump" % add_dot(dossier_df.shape[0]))
print("In total there are  %s documents in the current woogle dump" % add_dot(document_df.shape[0]))

# %% [markdown]
# To gain a bit more insight into these numbers, we also make KDE plots to so the distributions of pages in documents and documents in dossiers.

# %%
bodytext_df['foi_bodyText'].isna().mean() # missing aligned text
bodytext_df['foi_bodyTextOCR'].isna().mean() # missing ocr'd text
# %%
dossier_size = document_df.groupby('foi_dossierId').size()
document_size = document_df['foi_nrPages']

fig, axes = plt.subplots(nrows=1, ncols=2)
dossier_size.plot(kind='kde', ax=axes[0], logx=True)
document_size.plot(kind='kde', ax=axes[1], logx=True)

axes[0].set_xlabel("Number of documents")
axes[1].set_xlabel("Number of pages")

plt.suptitle(
    "Plots of the distribution of the number of documents and pages\n in the dossiers and documents respectively")
plt.tight_layout()
plt.show()
# %%
dossier_size_filtered = dossier_size[dossier_size < 100]
dossier_size_filtered = dossier_size_filtered[1 < dossier_size_filtered]
dossier_size_filtered.plot(kind='hist', bins=99, x='Number of documents', y='Number of dossiers')
plt.show()

# %% [markdown]
# <a id="missing_values" />

# %% [markdown]
# ## Missing values
# 
# Due to various reasons, there are entries in the dataframes that have missing values. After loading in the data, we will start by looking at these missing values in the dataframes per column.

# %%
# print missing values summary/ plot
bodytext_nans = bodytext_df.isna().mean()
document_nans = document_df.isna().mean()
dossier_nans = dossier_df.isna().mean()

# Filter out zeros
# bodytext_nans = bodytext_nans.sort_index()
# dossier_nans = dossier_nans.sort_index()

# %%
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
bodytext_nans.plot(kind='barh', title="Fraction of NaN (or NaT) values in each column of bodytext dataframe",
                   ax=axes[0])
dossier_nans.plot(kind='barh', title="Fraction of NaN (or NaT) values in each column of dossiers dataframe", ax=axes[1])
document_nans.plot(kind='barh', title="Fraction of NaN (or NaT) values in each column of document dataframe",
                   ax=axes[2])
plt.show()

# %% [markdown]
# For the `bodytext_dataframe`, the ,large amount of missing values ofr the OCR redaction columns is due to the fact that one category of documents (1b) are HTML pages, which do not have redaction info.
# 
# Another interesting observation to make is the difference between the `foi_bodyText` and `foi_bodyTextOCR`, where the former has much more missing values than the latter. This difference is due to the fact that the original page sometimes contain a layer of text that can be extracted using a tool such as `pdftotext`, but sometimes they do not in which case it is hard to extract text from this page. Because of this, we have added the `foi_bodytextOCR` column, which extracts the text from documents by using Optical Character Recognition (OCR) with [Tesseract](https://github.com/tesseract-ocr/tesseract).

# %% [markdown]
# <a id="analyses" />

# %% [markdown]
# ## More counts and analyses
# 
# 
# Now that we have looked at the NaN values we will dive a bit deeper and look at the actual information in the dataframe, starting with the date column.
# 
# 
# For more detailed information about the number of documents per provider and over which periods providers publish, please visit https://woogle.wooverheid.nl/overview , where you will automatically find an up-to-date overview of what is in the database.

# %%
dossier_df['foi_publishedDate'].dt.year.plot(kind='hist')
plt.xlabel("Year")
plt.title("Histogram of the distribution of dates on which\n dossiers where published")
plt.show()

# %% [markdown]
# Although the number of dossiers that has information on both the request and the decision date is limited, we can look at the average processing time, i.e. the time between the request and the decision. For this, it only really makes sense to look at the "Woo Verzoek" category, which has `dc_type` 2i, as these is the only type that has information on the request and decision dates.

# %%
delay = dossier_df[dossier_df.dc_type == '2i']['foi_decisionDate'] - \
        dossier_df[dossier_df.dc_type == '2i']['foi_requestDate']
delay = delay[~delay.isnull()]
print("The average processing time is %s days" % delay.mean().days)

# %%
delay.dt.days.plot(kind='kde')
plt.xlabel("Number of days between request and decision")
plt.title("Distribution of the processing time of a Woo Dossier")
plt.show()

# %% [markdown]
# <a id="joins" />

# %% [markdown]
# ## Joining the dataframes
# 
# We can view these dataframes separately, but it is of course more interesting to combine the data from the dataframes in order to arrive at new insights. The most interesting is probably an example of how we can link the text of the pages so that we can do analytics on the text of the pages, and then group them by year or provider. We can do this by joining on the indices of the `bodytext` and `document` dataframes, which now share the same identifiers. It is most convenient to first join the `file` and `document` dataframes, then we have the information of both the documents and the files at our disposal.

# %%
# join the dossier and document dataframes
dossier_and_document_dataframe = document_df.join(dossier_df, on='foi_dossierId', rsuffix="_dossier")
# %%
# count the number of documents per dossier
dossier_gb = dossier_and_document_dataframe.groupby('foi_dossierId').size()
# %%
len(dossier_gb[dossier_gb == 3]) / len(dossier_gb)
# dossier_gb[dossier_gb > 1]
# len(dossier_gb)
# %%
dossier_gb_new.plot(kind='hist', bins=100, x='Number of documents', y='Number of dossiers')
plt.xlim(0, 10)
plt.xlabel('Number of documents')
plt.title('Histogram of the number of documents in the dossiers')
plt.ylabel('Number of dossiers')
plt.show()
# %%
complete_dataframe = bodytext_df.join(dossier_and_document_dataframe)

# %% [markdown]
# Now that we have combined the information from the different dataframes, we can do all kinds of interesting things, for example, finding out how much redacted text is documents on average, grouped by provider.

# %%
most_redacted_suppliers = complete_dataframe.groupby('dc_publisher_name')['foi_percentageTextAreaRedacted'].apply(
    lambda x: x.dropna().mean()).dropna()

# %%
redacted_grouped_by_supplier = most_redacted_suppliers.sort_values(ascending=False).dropna()

# %%
redacted_grouped_by_supplier

# %%
redacted_grouped_by_supplier[::-1].plot(kind='barh', title="percentage of redacted text, grouped by supplier")
plt.tight_layout()
plt.show()

# %% [markdown]
# <a id = "documentation" />

# %% [markdown]
# ## Explanation of columns
# 
# Below is an elaborate description of all columns in all of the three dataframes. Where applicable the source of the numbers is also reported and the method of how the numbers were obtained.

# %% [markdown]
# ### Bodytext dataframe
# 
# As mentioned, the bodytext dataframe contains all the information on the pages in Woogle, with the OCRed text and information on the redacted text on the page.
# 
# **foi_pageNumber**
#  - This specifices the page number of a specific page within a document. The page numbers start at 1, and have been obtained by getting the length of the pdf through `pdfplumber` and using this to get the pagenumbers of respective pages.
# 
# **foi_bodyText**
#  - This is the body text of the page of a PDF. This has been obtained by running the `pdftotext` tool through the PDFplumber tool, where the layout was preserved.
# 
# **foi_bodyTextOCR**
#   - This is the text on the pdf page that has been obtained by running the open-source [Tesseract] Engine (version 4.0) using Sauvola binarization on 300dpi PNG images extracted from the original pdf. For the procedure, language models for Dutch and English text were used.
# 
# **foi_hasOCR**
#   - Boolean value indicating whether or not there is Tesseract OCRed text available for a specific page. There are several reasons why a page might not have any text from Tesseract, with the biggest reason being errors while processing the documents which prevents Tesseract from properly OCRing the documents.
# 
# The next few columns concern the detection of redacted text in the documents. This has been extracted using information from Tesseract, using the approach described in https://scripties.uba.uva.nl/search?id=record_52795).
# 
# **foi_redacted**
#    - Boolean value specifying whether there is redacted present on the page.
# 
# **foi_nrRedactedRegions**
#    - Integer specifying the number of individual regions of redacted text that are present on a page.
# 
# **foi_contourArea**
#    - float values specifying the total size (number of pixels) of redacted information in a page.
# 
# **foi_textArea**
#    - float specifying the number of pixels on a page which belong to text.
# 
# **foi_charArea**
#    - float specifying the number of pixels on a page that is part of a character.
# 
# **foi_percentageTextAreaRedacted**
#    - Float value specifying the percentage of the total texta area that is redacted
# 
# **foi_percentageCharAreaRedacted**
#    - Float value specifying the percentage of the total characters that is redacted

# %% [markdown]
# ### Dossiers Dataframe
# 
# In the case of a complete dossier, there are often multiple documents associated with the dossier. Typically there is a document containing the decision regarding the request, the text from the original request, and if released, the actual documents that were requested. Often there is also an index or 'inventaris', that contains a table/list of all the documents that are contained within the dossier.
# 
# 
# ## Dossiers
# 
# **dc_title**
#    - string containing the title of the dossier.
#     
# **dc_description**:
#    - string containing a short description of the dossier.
# 
# **dc_type**
#    - type of the dossier (e.g. 2i), these types concern for different types of requests.Currently supported types are 2b , 2d, 2i, 2k, 2f.
# 
# **dc_type_description**
#    - description of the dossier type (e.g. Dossier na Wob/Woo-verzoek).
#     
# **dc_publisher**
#    - publisher of the dossier, TOOI code (e.g. gm0363), a complete list of the current publishers can be found at
#     https://woogle.wooverheid.nl/overview
# 
# **dc_publisher_name**
#    - full name of the publisher (e.g. Gemeente Amsterdam) in plain text.
# 
# **dc_source**
#    - source of the dossier in the form of a URL that links to the original location from which the dossier was obtained. This is not always a direct link to the dossier but rather to a indexing page where multiple dossiers are listed in most cases.
# 
# **foi_publishedDate**
#    - date of publication of the dossier (this refers to the date on which the dossier was originally published, not when it was published via Woogle)
# 
# **dc_date_year**
#    - year of the dossier (This is usually the year in which te decision was made, and if this does not exist it wil lle be the date in which the dossier was released.)
# 
# **foi_requestDate**
#    - date of the request for the dossier, which has been extracted from the file using 
# 
# **foi_decisionDate**
#    - date of the decision concerning the release of the documents requested in the WOO request.
# 
# **foi_valuation**
#    - valuation of the dossier, which indicates the status of the documents, this can have 4 different values: 
#         
#         - Openbaar  
#            - All of the documents that were requested were made public
#         
#         - Deels openbaar
#            - A part of the documents that were requested were released, but not all
#         
#         - Geen documenten 
#            - No documents were released
#         
#         - Reeds openbaar
#            - The documents that were requested were already public.
# 
# **foi_requestText**
#    - text of the request of the document, which contains the original request made for the publication of the documents. This text has been extracted using `pdftotext`
# 
# **foi_decisionText**
#    - text of the decision on the document, which contains a file with a decision and explanation for the publication of the documents. This text has been extracted using `pdftotext`
# 
# **foi_isAdjourned**
#    - Boolean value indicationg whether or not the WOO request is adjourned.
# 
# **foi_requester**
#    - requester of the dossier, this is encoded in several categories that indicate who request the documents. This is mostly a legal term, indicating whether it was a person or an organization, although specific organizations are also sometimes given.
# 
# **foi_retrievedDate**
#    - date of retrieval of the dossier with the collection script if it was not manually added to Woogle.
# 

# %% [markdown]
# ### Documenten Dataframe

# %% [markdown]
# **dc_identifier**
#    - unique document identifier
#     
# **foi_dossierId**
#    - unique dossier identifier where document belongs to (as seen in the example above, this can be used to join the dataframes together)
#     
# **dc_title**
#    - string specyfing the title of the document.
#     
# **foi_fileName**
#    - String specifying the filename of the document.
#     
# **dc_format**
#    - format of the document (mime type). Currently there are four file formats in the dataframe, text/html, pdf files, spreadsheets and jpegs.
#     
# **dc_source**
#    - source of the document, this is a url that points to the original location where the document was retrieved from.
#     
# **dc_type**
#    - type of the document (e.g. 'bijlage' or 'besluit'), this can be either 'bijlage', 'besluit', 'verzoek' or 'inventaris', which we have described briefly above when discussing the woo dossiers.
#     
# **foi_nrPages**
#    - integer specifying number of pages in the document.
#
