"""
pong - The Artwork Recommendation System
========================================

License: MIT

This file contains 3rd level artwork general details category enum.
"""


from enum import Enum

from category_object import CategoryObject
from data_types import DataTypes


class ArtWorkGeneral(Enum):

    GENRE = CategoryObject(['artwork', 'general', 'genre'],
                           DataTypes.ENUM_ARRAY, 1)
    STYLE = CategoryObject(['artwork', 'general', 'style'],
                           DataTypes.ENUM_ARRAY, 0)
    THEME = CategoryObject(['artwork', 'general', 'theme'],
                           DataTypes.ENUM_ARRAY, 2)
