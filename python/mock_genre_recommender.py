"""
pong - The Artwork Recommendation System
========================================

License: MIT

This file contains a mock genre recommender prototype class.
"""


from artwork_general import ArtWorkGeneral
from data_structure_object import DataStructureObject
from processing_types import ProcessingTypes
from recommender import Recommender


class MockGenreRecommender(Recommender):
    """
    Provide a mock genre recommender
    ================================
    """

    def __init__(self):

        self._input_structure = [DataStructureObject(ArtWorkGeneral.GENRE,
                                                     3,
                                                     [ProcessingTypes
                                                      .FILL_ARRAY_WITH_ZEROS])]
        self._output_structure = []
        self._model = ''
        super().__init__()


    def predict(self, inputs):
        """
        Make predictions
        ================

        Parameters
        ----------
        inputs : any
            Parameters to use for prediction.

        Returns
        -------
        any
            Predicted values, aka. outputs.

        Notes
        -----
            This is a mock predictor so it predicts "Happy" and strength of 3.
        """

        return ('Happy', 3)
