"""
pong - The Artwork Recommendation System
========================================

License: MIT

This file contains recommender base class.
"""


from abc import abstractmethod


class Recommender:
    """
    Provide base recommender functionality
    ======================================
    """

    def __init__(self):
        """
        Initialize the object
        =====================
        """

        try:
            if not all([self._input_structure != None,
                        self._output_structure != None,
                        self._model != None]):
                raise RuntimeError('Recommender subclass init failed.')
        except AttributeError:
            raise RuntimeError('Recommender subclass init failed.')


    @abstractmethod
    def predict(self, inputs : any) -> any:
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
        """
