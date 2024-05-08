Preprocess a Woogle Dump
========================

A `Woogle dump <https://doi.org/10.17026/dans-zau-e3rk>`_ consists of the following files:

- ``woo_dossiers.csv``, which contains the metadata of the dossiers.
- ``woo_documents.csv``, which contains the metadata of the documents.
- ``woo_bodytext.csv``, which contains the body text of the documents.

To preprocess a Woogle dump, place these files in ``data/extracted`` and run ``split_woogle_dump_per_dossier.py``.
This creates individual ``.pkl`` files for each dossier with at least 20 documents in ``data/raw``.
This script is not very efficient, so it may take a few hours to run.

The paths and the minimum number of documents per dossier can be configured in ``conf/split_woogle_dump_per_dossier.yaml``.
