from gensim.models import LdaModel, LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import numpy as np
import pandas as pd
import nltk
import seaborn as sns
from nltk.corpus import stopwords
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import polars as pl
import time
import pickle
from lexrank import LexRank

# %%
start = time.time()
dossier = pd.read_csv('data/formatted_documents.csv')
bodies = pd.read_csv('data/raw/woo_bodytext.csv.gz')
end = time.time()
print("Done in {} seconds".format(end - start))


# %%
start = time.time()
dossier_pl = pl.read_csv('data/formatted_documents.csv')
bodies_pl = pl.read_csv('data/woo_bodytext.csv.gz')
end = time.time()
print("Done in {} seconds".format(end - start))

# %%
# Generate sample
body_sample = bodies.sample(n = 10000).reset_index(drop=False)
dossier_docs = dossier['full_text'].astype(str).tolist()
docs = body_sample['foi_bodyTextOCR']
docs = docs.astype(str).tolist()
# %%
# Preprocess documents
stop_words = set(stopwords.words('dutch'))
lemmatizer = WordNetLemmatizer()

def preprocess_documents(documents):
    preprocessed_docs = []
    for doc in documents:
        # Tokenize
        tokens = word_tokenize(doc.lower())
        # Remove stopwords and lemmatize
        filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w.isalpha()]
        preprocessed_docs.append(filtered_tokens)
    return preprocessed_docs

preprocessed_docs = preprocess_documents(docs)
dossier_processed = preprocess_documents(dossier_docs)

# %%
# Create dictionary and BoW
id2word = Dictionary(preprocessed_docs)
id2word.filter_extremes(no_below=20, no_above=0.5)
corpus = [id2word.doc2bow(text) for text in preprocessed_docs]

# %%
# Test different values
def optimize_lda_topics(corpus, id2word, preprocessed_docs, min_topics=15, max_topics=30, workers=31, random_state=42,
                        passes=1, alpha='symmetric', eta='symmetric'):
    coherence_scores = []
    topic_numbers = list(range(min_topics, max_topics + 1))

    for num_topics in topic_numbers:
        lda_model = LdaMulticore(corpus=corpus,
                                 id2word=id2word,
                                 num_topics=num_topics,
                                 workers=workers,
                                 random_state=random_state,
                                 passes=passes,
                                 alpha=alpha,
                                 eta=eta)

        coherence_model_lda = CoherenceModel(model=lda_model, texts=preprocessed_docs, dictionary=id2word,
                                             coherence='c_v')
        coherence_score = coherence_model_lda.get_coherence()
        coherence_scores.append(coherence_score)
        print(f"Num Topics: {num_topics}, Coherence Score: {coherence_score}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(topic_numbers, coherence_scores)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("Coherence Scores for Different Number of Topics")
    plt.xticks(topic_numbers)
    plt.grid(True)
    plt.show()

    # Finding the optimal number of topics
    max_coherence_score = max(coherence_scores)
    optimal_num_topics = topic_numbers[coherence_scores.index(max_coherence_score)]
    print(f"The optimal number of topics is {optimal_num_topics} with a coherence score of {max_coherence_score}.")

    return optimal_num_topics

optimize_lda_topics(corpus, id2word, preprocessed_docs) # ~ 15 or 28

# %%
lda_model = LdaModel(corpus=corpus,
                                 id2word=id2word,
                                 num_topics=14,
                                 random_state=42,
                                 passes=3,
                                 alpha="auto",
                                 eta="auto")

coherence_model_lda = CoherenceModel(model=lda_model, texts=preprocessed_docs, dictionary=id2word,
                                             coherence='c_v')
print(f'Final coherence model: {coherence_model_lda.get_coherence()}')

# %%
lda_model.save('tm/lda_model_v2')


# %%
expElogbeta_path = 'tm/lda_model_v2.expElogbeta.npy'
expElogbeta = np.load(expElogbeta_path)

print(f'The expElogbeta: {expElogbeta.shape}')
dict = Dictionary.load('tm/lda_model_v2.id2word')
print(f'The dictionary docs: {dict.num_docs}')
print(f'The dictionary tokens: {dict.num_pos}')

lda = LdaModel.load('tm/lda_model_v2')
print(f'The lda models topics: {lda.num_topics}')

# %%
# Optimization
def compute_coherence_values(corpus, dictionary, texts, num_topics_range, passes_range, alpha_range):
    results = []
    for num_topics in num_topics_range:
        for passes in passes_range:
            for alpha in alpha_range:
                for eta in eta_range:
                    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=passes, alpha=alpha, eta=eta)
                    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
                    coherence_score = coherence_model_lda.get_coherence()
                    results.append((num_topics, passes, alpha, coherence_score))
                    print(f'num_topics={num_topics}, passes={passes}, alpha={alpha}, eta = {eta}, Coherence={coherence_score}')
    return results

# Define the range for the parameters
num_topics_range = [15, 20, 25, 30]
passes_range = [10]
alpha_range = ['auto']
eta_range = ['auto']

# Compute coherence values for the various combinations of parameters
results = compute_coherence_values(corpus=corpus, dictionary=id2word, texts=preprocessed_docs, num_topics_range=num_topics_range, passes_range=passes_range, alpha_range=alpha_range)

# Find the parameters that give the highest coherence score
best_result = max(results, key=lambda x: x[3])
print(f'Best Model Parameters: num_topics={best_result[0]}, passes={best_result[1]}, alpha={best_result[2]}, Coherence={best_result[3]}')

# %%
# Compute dossier coherence
lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=best_result[0], random_state=42, passes=best_result[1], alpha=best_result[2], per_word_topics=True)
new_corpus = [id2word.doc2bow(doc) for doc in dossier_processed]
new_doc_topics = [lda_model[doc_bow] for doc_bow in new_corpus]

# %%
# Export model to file
lda_model.save('tm/lda_model_v1')

# %%
# Use JS or KL Divergence to find edge weights (only for topical distance)
def get_topic_distribution(lda_model, doc_bow):
    # Get the topic distribution for a document
    topic_distribution = np.zeros(lda_model.num_topics)
    for topic, prob in lda_model.get_document_topics(doc_bow, minimum_probability=0):
        topic_distribution[topic] = prob
    return topic_distribution

# Extract the topic distributions for the first 10 documents
topic_distributions = [get_topic_distribution(lda_model, doc) for doc in new_corpus]

# Calculate the JS divergence between each pair of documents
n_docs = len(topic_distributions)
adjacency_matrix = np.zeros((n_docs, n_docs))

for i in range(n_docs):
    for j in range(n_docs):
        # JS divergence is symmetric, so we calculate it for one half and mirror it
        if i != j:
            js_divergence = jensenshannon(topic_distributions[i], topic_distributions[j])**2
            adjacency_matrix[i, j] = js_divergence
            adjacency_matrix[j, i] = js_divergence

# %%
# Try spectral clustering
adj_mat = np.array(adjacency_matrix)
spec_clustering = SpectralClustering(n_clusters=3,
    assign_labels='discretize',
     random_state=0).fit(adj_mat)

cluster_labels = spec_clustering.labels_

silhouette_avg = silhouette_score(adjacency_matrix, cluster_labels, metric='precomputed')
print(f'Silhouette avg: {silhouette_avg}')

ch_avg = calinski_harabasz_score(adjacency_matrix, cluster_labels)
print("The average Calinski-Harabasz is : ", ch_avg)

db_avg = davies_bouldin_score(adjacency_matrix, cluster_labels)
print("The average Davies-Bouldin_score is : ", db_avg)

# %%
# Visualize
# Create a mapping from nodes to clusters
node_cluster_mapping = {node: cluster for node, cluster in enumerate(cluster_labels)}

# Sort the nodes by cluster
sorted_nodes = sorted(node_cluster_mapping, key=lambda node: node_cluster_mapping[node])

# Reorder the adjacency matrix
reordered_adjacency_matrix = adj_mat[sorted_nodes, :][:, sorted_nodes]

plt.figure(figsize=(10, 8))  # Set the size of the figure
heatmap = plt.imshow(reordered_adjacency_matrix, cmap='Reds', aspect='auto')

# Add a color bar to the side
plt.colorbar(heatmap)

# Add labels and titles as needed
plt.title('Reordered Adjacency Matrix Heatmap')
plt.xlabel('Index')
plt.ylabel('Index')

# Optionally, you can add ticks and labels for each row/column if the matrix size is manageable
# For a large matrix, you might want to skip this or use it sparingly
# plt.xticks(range(len(reordered_adjacency_matrix)), labels, rotation=45)  # 'labels' should be your custom labels
# plt.yticks(range(len(reordered_adjacency_matrix)), labels)

plt.show()
# %%
# Using hierarchical clustering with Ward distance (i.e. minimize within cluster variance)
linked = linkage(adjacency_matrix, 'ward')

plt.figure(figsize=(10, 7))
dendrogram(linked, labels=list(range(214)), orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Document Index')
plt.ylabel('Distance (Ward)')
plt.show()

# %%
# Cluster
# Let's assume you decided on a distance threshold from the dendrogram, e.g., 5
distance_threshold = 5
cluster_labels = fcluster(linked, distance_threshold, criterion='distance')

# Print clusters for each document
for i, cluster_id in enumerate(cluster_labels):
    print(f"Document {i}: Cluster {cluster_id}")

# %%
# Measure cluster quality
tsne = TSNE(n_components=2, metric="precomputed", random_state=42, init='random')
docs_2d = tsne.fit_transform(adjacency_matrix)

silhouette_avg = silhouette_score(docs_2d, cluster_labels)
print("The average silhouette_score is :", silhouette_avg)

ch_avg = calinski_harabasz_score(docs_2d, cluster_labels)
print("The average Calinski-Harabasz is : ", ch_avg)

db_avg = davies_bouldin_score(docs_2d, cluster_labels)
print("The average Davies-Bouldin_score is : ", db_avg)
