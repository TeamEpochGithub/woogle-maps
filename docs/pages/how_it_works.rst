How It Works
================

Most of our system is based on the work of :cite:t:`keith2021narrative`.
We assume that you are familiar with the concepts of the narrative map and the methodology described in this paper.

As a general overview, this image describes our data pipeline:

.. image:: ../../_static/pipeline.png
   :width: 800
   :alt: Woogle Maps - Pipeline

Preprocessing Pipeline
----------------------

Before we talk about the individual pipeline steps, we will first explain the pipeline itself.

The pipeline we use inherits from :py:class:`epochalyst.pipeline.model.transformation.transformation.TransformationPipeline`
from `Epochalyst <https://github.com/TeamEpochGithub/epochalyst>`_, Team Epoch's basis for AI competitions.
This pipeline runs the data through all the steps we described above with functionalities such as caching intermediate results and logging already implemented.
All the steps inherit from :py:class:`epochalyst.pipeline.model.transformation.transformation_block.TransformationBlock`.
This allows each step to be developed independently and to be easily added or removed from the pipeline for quick experimentation.

This pipeline is implemented in in :py:class:`src.pipeline.verbose_transformation_pipeline.VerboseTransformationPipeline`.

Preprocessing Pipeline Blocks
-----------------------------

PDF to Text
+++++++++++

This step is only necessary if the input is PDF files.
We parse the files to extract the ``date`` from the metadata, the ``title`` from the filename, and ``full_text`` from the document itself.
This data is stored in a pandas DataFrame.

This functionality is implemented in :py:mod:`src.preprocessing.pdf_to_text`.


Extract Dates
+++++++++++++

For documents that don't have a date, we extract the date from the title or the text.
We currently do this using a regex that matches texts for dates in the format ``dd-mm-yyyy`` or ``dd monthname yyyy``.
For example, it will find ``12-12-2023`` or ``12 augustus 2023``.
If it finds multiple dates, it assumes the first one is the creation date.
If it doesn't find any dates, the date remains ``pd.NaT``.

This functionality is implemented in :py:mod:`src.preprocessing.extract_dates_regex`.

We chose this approach because it is simple and works well for most of the documents we have,
as most letters and reports have a date in the title or in the header of the document.
In some cases, the date in the header is unreadable, or the actual creation date is mentioned later in the document,
in which case this approach does not work.
For future work, an NLP can be used to extract the date from the text instead. See :doc:`limitations` for more information.


Summarize Documents
+++++++++++++++++++

Since documents can vary wildly in length, we summarize them to somewhat normalize the length.
We use a `LexRank <https://pypi.org/project/lexrank/>`_ model to find the most important sentences in each text.
This summary gets stored in the ``summary`` column.

.. warning::
   Documentation is still WIP:

   - Add more about the LexRank model.
   - Explain why we chose this approach.

This functionality is implemented in :py:mod:`src.preprocessing.extract_important_sentences`.


Find Common Topics
++++++++++++++++++

We use an `Gensim <https://radimrehurek.com/gensim/>`_ `Latent Dirichlet Allocation (LDA) <https://radimrehurek.com/gensim/models/ldamodel.html>`_ model to find the most common topics between documents.
A model trained on the entire Woogle dataset is included in ``tm/`` in this repo.
You can train your own model using the ``notebooks/compute_topical_similarity.py`` notebook.
This model is used to find the topical distribution of each document based on all the topics it has found during training.

Our approach involves using Latent Dirichlet Allocation (LDA) to find the most common topics in the dataset.
This method is well-researched and widely used in the field of NLP, and it leverages the fact that documents that are similar in content are likely to be similar in topic as well.
Therefore, it assigns certain words to certain topics, and uses these probability distributions to find the most suitable topics for each document.
.. note:: LDA is commonly used for topic modelling, when the topics are not available and need to be inferred from the documents. Otherwise, a form of guided topic modelling is used to extract previously defined topics.

This functionality is implemented in :py:mod:`src.preprocessing.compute_topical_distribution`.


Random Walk Embedding
+++++++++++++++++++++

.. warning::
   Documentation is still WIP:

   - Add more about the Random Walk Embedding.
   - Explain why we chose this approach.

This functionality is implemented in :py:mod:`src.preprocessing.random_walk_embedding`.


Impute Missing Dates
++++++++++++++++++++

We impute the missing dates using the embeddings we created in the previous step.
It copies the date of the most cosine-similar document to the document with the missing date.
This is necessary, as all steps after this require all documents to have a date.

This functionality is implemented in :py:mod:`src.preprocessing.impute_dates`.


Cluster Documents
+++++++++++++++++

.. warning::
   Documentation is still WIP:

   - Add more about the Clustering.
   - Explain why we chose this approach.


This functionality is implemented in :py:mod:`src.preprocessing.cluster_documents`.


Linear Programming
++++++++++++++++++

.. warning::
   Documentation is still WIP:

   - Add more about the Linear Programming.
   - Explain why we chose this approach.


This functionality is implemented in :py:mod:`src.preprocessing.linear_programming`.


Create Events
+++++++++++++

Events represent the most crucial moments in the generated timelines.
Therefore, we create events by finding the most important clusters in the timeline.
Using the previously computed clusters, and the adjacency list for each document, we find the most similar documents within a cluster to produce an event.
These events are indicated later in the visualization.


This functionality is implemented in :py:mod:`src.preprocessing.create_events`.


Summarize Cluster
+++++++++++++++++

For explainability, we also create a summary of each cluster with the same technique we use for creating a summary of each document.
We use the same LexRank model as in the `Summarize Documents`_ step to find the most important sentences in each cluster.

This functionality is implemented in :py:mod:`src.preprocessing.cluster_explainer`.


Find Storylines
+++++++++++++++

We find the storylines by creating a graph using the clusters we created in the previous step.
We use the adjacency matrix found during `Create Events`_ to create a `RustworkX <https://www.rustworkx.org/>`_ graph.
We find storylines by, starting from the earliest cluster, finding the shortest path to all other clusters and taking the longest shortest path.
The main storyline is the first one we find, starting from the earliest cluster.
This approach is was also used in Keith's Narrative Maps, so we already know it works well.

This functionality is implemented in :py:mod:`src.preprocessing.find_storylines`.


Filter Redundant Edges
++++++++++++++++++++++

After finding the storylines, there are a lot of redundant edges the graph that we do not need anymore.
We perform transitive reduction and filter interstory connections the same way as in Keith's Narrative Maps.

This functionality is implemented in :py:mod:`src.preprocessing.filter_redundant_edges`.


Compute Layout
++++++++++++++

To render the graph, we need to compute the position of each cluster first.
By default, the clusters in the main storyline are placed at the vertical center of the screen, uniformly spaced horizontally.
The other storylines are placed above and below the main storyline, with the same horizontal spacing.

There is also an option for scaling the distance between the clusters based on the average date of the clusters.
This could make it easier to work with the timelines as it is more intuitive to have the distance between the clusters represent the time between them,
but a disadvantage of this is that clusters close to each other in time will be very close together on the screen, which could make it hard to read the text.
This option was planned to be toggled in the UI, but was not finished due to time constraints.

This functionality is implemented in :py:mod:`src.preprocessing.compute_layout`.


Rendering the Result
--------------------

After the pipeline has run, we still need to render the resulting map in the browser.
We use `Dash Cytoscape <https://dash.plotly.com/cytoscape>`_ for this.
We create elements for each storyline and for each cluster, and we create edges between the clusters.

This functionality is implemented in :py:mod:`dashapp.generate_graph_elements`.


Bibliography
------------

.. bibliography::
