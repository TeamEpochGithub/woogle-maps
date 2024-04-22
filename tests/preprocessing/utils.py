"""Utils for tests."""

import pandas as pd


def get_test_df() -> pd.DataFrame:
    """Get a small test dataframe."""
    bodytext_data = [
        {
            "foi_documentId": "doc1",
            "foi_pageNumber": 1,
            "foi_bodyText": "Text content of document 1",
            "foi_bodyTextOCR": "OCR content of document 1",
            "foi_hasOCR": True,
            "foi_redacted": 0.1,
            "foi_contourArea": 100.0,
            "foi_textArea": 200.0,
            "foi_charArea": 150.0,
            "foi_percentageTextAreaRedacted": 5.0,
            "foi_percentageCharAreaRedacted": 3.0,
        },
        {
            "foi_documentId": "doc2",
            "foi_pageNumber": 2,
            "foi_bodyText": "Text content of document 2",
            "foi_bodyTextOCR": "OCR content of document 2",
            "foi_hasOCR": False,
            "foi_redacted": 0.2,
            "foi_contourArea": 150.0,
            "foi_textArea": 250.0,
            "foi_charArea": 180.0,
            "foi_percentageTextAreaRedacted": 4.0,
            "foi_percentageCharAreaRedacted": 2.0,
        },
    ]

    document_data = [
        {
            "dc_identifier": "doc1",
            "foi_dossierId": "dossier1",
            "dc_title": "Document Title 1",
            "foi_fileName": "File1.pdf",
            "dc_format": "PDF",
            "dc_source": "Source 1",
            "dc_type": "Type 1",
            "foi_nrPages": 10.0,
        },
        {
            "dc_identifier": "doc3",
            "foi_dossierId": "dossier2",
            "dc_title": "Document Title 2",
            "foi_fileName": "File2.pdf",
            "dc_format": "PDF",
            "dc_source": "Source 2",
            "dc_type": "Type 2",
            "foi_nrPages": 20.0,
        },
    ]

    bodytext_df = pd.DataFrame(bodytext_data)
    document_df = pd.DataFrame(document_data)
    return document_df.merge(bodytext_df, left_on="dc_identifier", right_on="foi_documentId", how="left").rename(columns={"foi_bodyTextOCR": "full_text"})
