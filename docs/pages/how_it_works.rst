How It Works
================

.. warning::
   WIP: This section is still a work in progress.

Most of our system is based on a previous research paper called “Narrative Maps” from 2021 by Norambuena et al.
As input, we get the pre-selection of documents as PDF files.
We first find the creation dates from the title and text of each document
and we create a summary of each document by finding the most important sentences.
We also find the common topics between documents.
We then use all of this to create a numeric representation of the text.
This numeric representation can be used to “measure” how similar documents are.
After that we cluster, or group documents,
and for explainability, we also create a summary of each cluster with the same technique we use for creating a summary of each document.
We finally create a graph using those clusters, which just means we find how strong the relationship is between the clusters.
We need that graph to find all the storylines including the main one.
In the web app, we render the graph as a narrative map as you’ve seen before.
