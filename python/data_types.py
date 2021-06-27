"""
pong - The Artwork Recommendation System
========================================

License: MIT

This file contains enum of data types.
"""


from enum import Enum


class DataTypes(Enum):
    """
    Provide list of acceptable data types
    =====================================
    """

    ENUM = 'enum'                   # Simple classification
    ENUM_ARRAY = 'enum []'          # Multi-class classification
    FLOAT = 'float'                 # Also double, long double
    FLOAT_ARRAY = 'float []'        # Also array of doubles or long doubles
    INT = 'int'                     # Also any type of ints
    INT_ARRAY = 'int []'            # Also array of any types of ints
    NON_DATA = 'null'               # Non data content, like upper level
    OBJECT = 'object'               # Use only if no other type is useable
    OBJECT_ARRAY = 'object_array'   # Use only if no other type is useable
    TEXT = 'text'                   # Potential NLP input
