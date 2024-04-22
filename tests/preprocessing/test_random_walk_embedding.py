# """test the embeddings based on the random walk."""
#
# from unittest import TestCase, main
#
# import numpy as np
# import pandas as pd
#
# from src.preprocessing.random_walk_embedding import RandomWalkEmbedding
#
# # generate the dates and the topical similarities of 10 document
# dates = pd.date_range(start="2024-01-01", end="2024-01-11", freq="D")
# similarities = np.random.default_rng().random((11, 11)).tolist()
#
# # create the panda dataframe based on the date and similarity
# data = pd.DataFrame({"date": dates, "topics similarity": similarities})
#
#
# class RandomEmbeddingTest(TestCase):
#     """Test the random walk embedding class."""
#
#     deep_walk = RandomWalkEmbedding(threshold=0.6, num_walks=500, walk_length=30)
#
#     def test_date(self) -> None:
#         """Test if the date similarity computes the correct similarities."""
#         similarity = self.deep_walk.date_similarity(data)
#         assert np.isclose(similarity[0, 1], 1 / (1 + 0.1), rtol=1e-5)
#
#         assert np.isclose(similarity[4, 6], 1 / (1 + 0.2), rtol=1e-5)
#
#         assert np.isclose(similarity[4, 7], 1 / (1 + 0.3), rtol=1e-5)
#
#
# if __name__ == "__main__":
#     main()
