"""
Dataset processors for various model training tasks.
"""

try:
    from .text_processors import PROCESSOR_MAP as TEXT_PROCESSOR_MAP
except ImportError:
    TEXT_PROCESSOR_MAP = {} 