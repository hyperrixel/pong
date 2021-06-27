"""
pong - The Artwork Recommendation System
========================================

License: MIT

This file contains category object class.
"""


from data_types import DataTypes
from gdpr_filters import GDPRFilters


class CategoryObject:
    """
    Provide data object functionality
    =================================

    Attributes
    ----------
    data_type : DataTypes (read-only)
        Type identifier of the category.
    gdpr_filter : GDPRFilters (read-only)
        The GDPR filter of the category.
    has_id : bool (read-only)
        True if the category has AI inclusion ID, False if not.
    id : int (read-only)
        ID to use for the category in case of AI inclusion.
    taxonomy : list (read-only)
        List of the taxonomy eleents of the category.
    taxonomy_level : int
        The depth of the taxonomy of the category.
    taxonomy_string : str
        Taxonomy eleents of the category as a string.
    """

    def __init__(self, taxonomy : list,
                 data_type : DataTypes = DataTypes.NON_DATA,
                 data_point_id : int = -1,
                 gdpr_filter : GDPRFilters = GDPRFilters.NON_SENSITIVE):
        """
        Initialize the object
        =====================

        Parameters
        ----------
        taxonomy : list
            List of taxonomy tags. Tags should match the taxonomy structure.
        data_type : DataTypes, optional (DataTypes.NON_DATA if omitted)
            Type of the data.
        data_point_id : int, optional (-1 if omitted)
            ID, placement to use in poistioning at data structure composition.
            -1 stands for data that is not a data point data.
        gdpr_filter : GDPRFilters, optional
                                   (GDPRFilters.NON_SENSITIVE if omitted)
            GDPR filter characteristics.
        """

        self.__taxonomy = taxonomy
        self.__data_type = data_type
        self.__id = data_point_id
        self.__gdpr_filter = gdpr_filter


    @property
    def data_type(self) -> DataTypes:
        """
        Get data type of the category
        =============================

        Returns
        -------
        DataTypes
            Type identifier of the category.
        """

        return self.__data_type


    @property
    def gdpr_filter(self) -> GDPRFilters:
        """
        Get GDPR filter of the category
        ===============================

        Returns
        -------
        GDPRFilters
            The GDPR filter of the category.
        """

        return self.__gdpr_filter


    @property
    def has_id(self) -> bool:
        """
        Get whether the data point has AI inclusion ID or not
        =====================================================

        Returns
        -------
        bool
            True if the category has AI inclusion ID, False if not.
        """

        return self.__id != -1


    @property
    def id(self) -> int:
        """
        Get AI inclusion ID of the category
        ===================================

        Returns
        -------
        int
            ID to use for the category in case of AI inclusion.
        """

        return self.__id


    @property
    def taxonomy(self) -> list:
        """
        Get the taxonomy of the category
        ================================

        Returns
        -------
        list
            List of the taxonomy eleents of the category.
        """

        return self.__taxonomy.copy()


    @property
    def taxonomy_level(self) -> int:
        """
        Get the level in the taxonomy of the category
        =============================================

        Returns
        -------
        list
            The depth of the taxonomy of the category.
        """

        return len(self.__taxonomy)


    @property
    def taxonomy_string(self) -> str:
        """
        Get the taxonomy string of the category
        =======================================

        Returns
        -------
        list
            Taxonomy elements of the category as a string.
        """

        return '.'.join(self.__taxonomy)
