"""
pong - The Artwork Recommendation System
========================================

License: MIT

This file contains top level category enum.
"""


from enum import Enum

from category_object import CategoryObject


class CategoryMain(Enum):

    ARTWORK = CategoryObject(['artwork'])
    POSITIONAL = CategoryObject(['positional'])
    PERSONAL = CategoryObject(['personal'])
