"""
PRIMVS-Pipeline: Variable Star Catalogue Construction Pipeline

This package provides a clean, portable pipeline for constructing the PRIMVS
(PeRiodic Infrared Multiclass Variable Stars) catalogue from VIRAC time-series data.

Main Components:
    - data_access: Interface to VIRAC database
    - preprocessing: Data quality filtering
    - features: Statistical feature calculation
    - fap: False Alarm Probability calculation
    - aggregation: Catalogue construction
    - utils: Utility functions

Example:
    >>> from primvs_pipeline import Pipeline
    >>> from primvs_pipeline.config import load_config
    >>> 
    >>> config = load_config('config/pipeline_config.yaml')
    >>> pipeline = Pipeline(config)
    >>> results = pipeline.process_sources(source_ids)
"""

__version__ = "2.0.0"
__author__ = "Niall Miller"

from .pipeline import Pipeline
from .config import load_config

__all__ = ["Pipeline", "load_config", "__version__"]

