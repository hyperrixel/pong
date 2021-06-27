"""
pong - The Artwork Recommendation System
========================================

License: MIT

This file contains 2nd level artwork category enum.
"""


from enum import Enum

from category_object import CategoryObject


class CategoryArtwork(Enum):

    COLOR = CategoryObject(['artwork', 'color'])
    CONNECTED_EMOTIONS = CategoryObject(['artwork', 'connected_emotions'])
    CONTENT = CategoryObject(['artwork', 'content'])
    GENERAL = CategoryObject(['artwork', 'general'])
    METHOD = CategoryObject(['artwork', 'method'])
    PERSPECTIVE_COMPOSITION = CategoryObject(['artwork',
                                              'perspective_composition'])
