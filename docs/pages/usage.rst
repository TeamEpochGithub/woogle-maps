Usage
================

Once you have Woogle Maps set up as described in the :doc:`readme_link`, you will see the home screen:

.. image:: ../../_static/screenshots/home.png
   :width: 800
   :alt: Woogle Maps - Home

You can select one of your preprocessed dossiers in the dropdown menu.
If you want to preprocess a new dossier, you can upload your own documents.

.. note::
   The following file types are supported:
    - ZIP files with PDFs at the root level
    - PDF files
    - CSV files with columns "title", "date", and "full_text" where each row is one document

   Make sure that you upload at least 5 documents.

After that, hit the "Generate Narrative Map" button.
This step may take a few minutes, depending on the size of your dossier and the performance of your machine.

Once it is done processing your dossier, you will see the narrative map:

.. image:: ../../_static/screenshots/map.png
   :width: 800
   :alt: Woogle Maps - Map


Each of the grey rectangular containers describe a storyline within your dossier.
Each node in a storyline is a cluster of documents that are similar to each other.
The nodes are already sorted where to the left are the oldest documents and to the right are the latest documents.
The arrows show how the nodes are connected, both to other nodes within the storyline as well as nodes in other storylines.
The blue dashes arrows represent the main story line, the one with the most important documents.

You can click on a node to see what documents are included in that node.
A popup will appear with the date and title of the documents as well as an extractive summary.

.. image:: ../../_static/screenshots/cluster_info.png
   :width: 400
   :alt: Woogle Maps - Cluster Info
