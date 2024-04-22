"""Layout for the map page."""

from dash_extensions.enrich import dash, html

from dashapp.conf import FINAL_DATA_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH
from dashapp.generate_graph_elements import create_graph
from src.pipeline.verbose_transformation_pipeline import VerboseTransformationPipeline
from src.preprocessing.cluster_documents import ClusterDocuments
from src.preprocessing.cluster_explainer import ClusterExplainer
from src.preprocessing.compute_layout import ComputeLayout
from src.preprocessing.compute_topical_distributions import TopicalDistribution
from src.preprocessing.create_events import CreateEvents
from src.preprocessing.extract_dates_regex import ExtractDatesRegex
from src.preprocessing.extract_important_sentences import ExtractImportantSentences
from src.preprocessing.filter_redundant_edges import FilterRedundantEdges
from src.preprocessing.find_storylines import FindStorylines
from src.preprocessing.impute_dates import ImputeDates
from src.preprocessing.linear_programming import LinearProgramming
from src.preprocessing.random_walk_embedding import RandomWalkEmbedding

dash.register_page(__name__, path_template="/map/<dossier_id>", title="Woogle Maps")


def layout(dossier_id: str | None = None, clusters: str | None = None) -> dash.development.base_component.Component:
    """Create the layout for the map page.

    :param dossier_id: the dossier id.
    :param clusters: whether to cluster the documents.
    :return: the layout for the map page.
    """
    if dossier_id is None:
        return html.Title("Please provide a dossier id")

    steps = [
        ExtractDatesRegex(min_date="1950-01-01", max_date="2025-12-31"),
        ExtractImportantSentences(),
        TopicalDistribution(pretrained_model_name_or_path="./tm/lda_model_v3", dictionary_name_or_path="./tm/lda_model_v3.id2word"),
        RandomWalkEmbedding(
            threshold=15,
            num_walks=50,
            walk_length=120,
            dimension=120,
        ),
        ImputeDates(),
        ClusterDocuments(periods=4),
        LinearProgramming(min_cover=0.2, K=10, threshold=0.01),
        CreateEvents(period=4),
        ClusterExplainer(threshold=10),
        FindStorylines(),
        FilterRedundantEdges(),
        ComputeLayout(spacing_within_story="uniform", transpose=False),
    ]

    if clusters is not None:
        del steps[7]

    pipeline = VerboseTransformationPipeline(steps=steps, title="Uploaded Dossier Pipeline")
    data = pipeline.run_dossier_pipeline(dossier_id=dossier_id, raw_data_path=RAW_DATA_PATH, processed_data_path=PROCESSED_DATA_PATH, final_data_path=FINAL_DATA_PATH)

    return create_graph(data)
