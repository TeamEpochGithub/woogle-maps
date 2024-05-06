Limitations & Future Work
=========================

We want to emphasize that our project is not a finished product;
it is a `Minimum Viable Product <https://en.wikipedia.org/wiki/Minimum_viable_product>`_.
It has all the basic functionality you can expect, but not much more.

In this section we describe the limitations of our project and suggest some ideas for future work.

Lack of good Dutch Natural Language Processing (NLP) model
----------------------------------------------------------

Our main constraint was the lack of a good Dutch NLP Model, which is used to interpret text data.
This model could be used to improve the *creation date extraction*, *the clustering of the documents*,
*the summarization of the documents*, and the *storyline finding*.

.. warning::
   WIP: Describe more about:

   - Dutch NLP models we tried.
   - How we decided that they were not good enough.

A lot of the open Dutch models simply did not perform that well in our tests.
We tried the following models:
 * `RobBERT: Dutch RoBERTa-based Language Model <https://huggingface.co/pdelobelle/robbert-v2-dutch-base>`_

Keep in mind that we were not allowed to use closed source models such as `GPT-4 <https://openai.com/index/gpt-4>`_,
which would perform a lot better, but pose a potential privacy risk.
Training our own model would also not be a viable solution as that would be be an entire project in and of itself.

A few models, such as `Llama 3 8B - Dutch <https://huggingface.co/ReBatch/Llama-3-8B-dutch>`_ were published right after we completed this project
and naturally, more models are to come in the future.
These models are very likely to perform better than the ones we tested, so it is worth trying them out.

Named Entity Recognition (NER)
+++++++++++++++++++++++++++++++

Another functionality that can be achieved with a good Dutch NLP model is Named Entity Recognition (NER).
With that, you would be able to follow a specific person through the different storylines,
so you can see how they are involved in the different documents.

For this use case specifically, the Woogle dataset would not help any model training as most named have been censored.

Evaluation of Maps
------------------------

There is currently no automated way to measure how “good” a map actually is.
:cite:t:`keith2021narrative` evaluated their narrative maps by letting humans rate some relatively simple maps in a questionnaire.
This is not only very labour intensive, but also not viable for a dossier with over 200 documents.
Further research is needed to find a way to automatically evaluate the quality of a narrative map.


UI Functionalities
------------------

Another major constraint was time; we only had 8 weeks to create a solution,
and our focus was on the very technical parts that only we could make.
This resulted in many useful UI functionalities being scrapped.

For example, clicking on each cluster would open a view of map on how the documents within the cluster are related to each other.

Another relatively simple feature that would greatly help understand dossiers would be view the documents as PDF files
when the corresponding nodes are clicked.
Right now, you only see the file names and you have to match it with the PDF file name yourself if you want to actually see the contents.


Bibliography
------------

.. bibliography::
