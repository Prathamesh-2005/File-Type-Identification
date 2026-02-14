"""
Utility functions for the file type classification system
"""

# This file makes the utils directory a Python package
from .header_footer_detector import (
    detect_header,
    detect_footer,
    remove_header,
    remove_footer,
    remove_header_footer,
    analyze_file
)

__all__ = [
    'detect_header',
    'detect_footer',
    'remove_header',
    'remove_footer',
    'remove_header_footer',
    'analyze_file'
]
