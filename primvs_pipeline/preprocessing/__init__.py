"""
Preprocessing Module

Provides data quality filtering and cleaning for lightcurves.

This module applies various quality filters to ensure only high-quality
photometric data is used in the pipeline.
"""

from .quality_filters import QualityFilter, apply_quality_filters

__all__ = ["QualityFilter", "apply_quality_filters"]

