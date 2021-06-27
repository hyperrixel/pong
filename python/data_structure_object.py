"""
pong - The Artwork Recommendation System
========================================

License: MIT

This file contains data structure object class.
"""


class DataStructureObject:
    """
    Provide data srturcture object framework
    ========================================

    Attributes
    ----------
    data_object : any (read only)
        The type of the data object.
    length : int (read only)
        The length of the data in processed form.
    processing : list (read only)
        List of the IDs of the applied processings.
    """

    def __init__(self, data_object : any, length : int,
                 processing : list):
        """
        Initialize object
        =================

        data_object : any,
            Type of data object to store.
        length : int
            Length of the data.
        processing : ProcessingTypes
            Type of processing of the data.
        """

        self.__data_object = data_object
        self.__length = length
        self.__processing = processing


    @property
    def data_object(self) -> any:
        """
        Get data object
        ===============

        Returns
        -------
        any
            The type of the data object.
        """

        return self.__data_object


    @property
    def length(self) -> int:
        """
        Get length of the data
        ======================

        Returns
        -------
        int
            The length of the data in processed form.
        """

        return self.__length


    @property
    def processing(self) -> list:
        """
        Get data processing workflow ID
        ===============================

        Returns
        -------
        list[ProcessingTypes]
            List of the IDs of the applied processings.
        """

        return self.__processing
