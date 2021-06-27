"""
pong - The Artwork Recommendation System
========================================

License: MIT

This file contains a short mock workflow.
"""


from mock_genre_recommender import MockGenreRecommender


# Use-case: we should build a basic recommendation system that uses artwork's
#           genre as the only input feature in one-hot-encoded form and outputs
#           a single emoji class with its strengthness.

data_record = [10]   # This entry has only 1 genre and its ID is 10.
predictor = MockGenreRecommender()
prediction = predictor.predict(data_record)
print(prediction)
