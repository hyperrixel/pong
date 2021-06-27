"""
pong - The Artwork Recommendation System
========================================

License: MIT

This file contains processing types enum.
"""


from enum import Enum

class ProcessingTypes(Enum):
    """
    Provide processing type identifiers
    ===================================
    """

    NO_PROCESSING = 0
    ONE_HOT_ENCODE = 1
    RATE_ENCODE = 2
    FILL_ARRAY_WITH_ZEROS = 3
    FILL_ARRAY_WITH_FIRST = 4
    FILL_ARRAY_WITH_LAST = 5
