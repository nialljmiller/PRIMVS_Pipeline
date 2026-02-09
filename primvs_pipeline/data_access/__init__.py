"""
Data Access Module

Provides interfaces for accessing VIRAC lightcurve data and metadata.

This module abstracts away the details of the VIRAC database structure,
providing a clean API for retrieving lightcurves and associated metadata.
"""

from .virac_interface import ViracInterface, load_lightcurve
from .file_io import save_catalogue, load_catalogue

__all__ = ["ViracInterface", "load_lightcurve", "save_catalogue", "load_catalogue"]

