import pandas as pd

from src.preprocessing.compute_layout import ComputeLayout


def check_storyline_y_alignment(data: pd.DataFrame):
    """Check if the Y coordinates are aligned within the same storyline."""
    for name, group in data.groupby("storyline"):
        if group["y"].nunique() != 1:
            return False
    return True

def test__compute_layout():
    data = pd.DataFrame(
        {
            "adj_list": [[8, 10, 1], [7, 8, 2], [7, 8, 10], [4, 5], [7, 8, 6], [7], [8, 9, 10], [8, 10], [12, 10, 9], [12, 13, 10], [11, 13], [13], [], []],
            "adj_weights": [
                [0.0640658152988518, 0.06406582718181346, 0.8718683575193347],
                [0.11111110958452239, 0.11111110958452239, 0.7777777808309553],
                [0.3333333127243863, 0.3333333127243863, 0.3333333745512273],
                [0.6666666666666666, 0.3333333333333333],
                [0.25, 0.25, 0.5],
                [1.0],
                [0.25, 0.25, 0.5],
                [0.09999999958782103, 0.900000000412179],
                [0.19999999999999998, 0.2000000247307379, 0.5999999752692621],
                [0.33333331959403506, 0.33333331959403506, 0.3333333608119299],
                [0.39835457951843506, 0.6016454204815649],
                [1.0],
                [],
                [],
            ],
            "storyline": [0, 0, 0, 1, 1, 2, 3, 2, 1, 3, 0, 4, 1, 0],
        },
    )
    block = ComputeLayout()
    res = block.transform(data)

    assert check_storyline_y_alignment(res)

def test__compute_layout__with_clusters():
    data = pd.DataFrame({
        "title": [
            "Covid in Wuhan",
            "Sars ruled out",
            "First death reported",
            "Lockdown in New York",
            "New strain found",
            "Vaccine development begins",
            "Global cases rise",
            "Travel restrictions applied",
        ],
        "date": [
            "2023-02-19", "2023-02-24", "2023-02-28", "2023-03-21",
            "2023-03-22", "2023-05-02", "2023-04-12", "2023-05-17",
        ],
        "clusters": [
            0, 0, 0, 1, 1, 1, 2, 2,
        ],
        "adj_list": [
            [1, 2],
            [1, 2],
            [1, 2],
            [0],
            [0],
            [0],
            [0, 1],
            [0, 1],
        ],
        "adj_weights": [
            [0.7, 0.3],
            [0.7, 0.3],
            [0.7, 0.3],
            [0.5],
            [0.5],
            [0.5],
            [0.6, 0.4],
            [0.6, 0.4],
        ],
        "storyline": [0, 0, 0, 1, 1, 1, 0, 0]
    })

    block = ComputeLayout()
    res = block.transform(data)

    assert check_storyline_y_alignment(res)
