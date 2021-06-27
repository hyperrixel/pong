"""
pong - The Artwork Recommendation System
========================================

License: MIT

This file contains enum for GDPR filters.
"""


from enum import Enum


class GDPRFilters(Enum):
    """
    Provide GDPR filters
    ====================
    """

    NON_SENSITIVE = 'non_sensitive'
    SENSITIVE = 'sensitive'
