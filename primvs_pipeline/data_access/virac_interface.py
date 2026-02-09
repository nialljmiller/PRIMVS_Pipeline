"""
VIRAC Database Interface

Provides clean access to VIRAC lightcurve data.

This module handles all interactions with the VIRAC FITS database,
extracting lightcurves and metadata for variable star analysis.

Author: Niall Miller (refactored)
Date: 2025-10-21
"""

import numpy as np
from astropy.io import fits
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ViracInterface:
    """
    Interface to VIRAC lightcurve database.
    
    Provides methods to retrieve lightcurves and metadata from VIRAC FITS files.
    
    Attributes:
        lc_dir: Directory containing lightcurve FITS files
        meta_dir: Directory containing metadata tables
    """
    
    def __init__(self, lc_dir: str, meta_dir: Optional[str] = None):
        """
        Initialize VIRAC interface.
        
        Args:
            lc_dir: Path to directory containing lightcurve FITS files
            meta_dir: Path to directory containing metadata tables (optional)
        """
        self.lc_dir = Path(lc_dir)
        self.meta_dir = Path(meta_dir) if meta_dir else None
        
        if not self.lc_dir.exists():
            logger.warning(f"Lightcurve directory does not exist: {self.lc_dir}")
        
        logger.info(f"Initialized VIRAC interface with LC dir: {self.lc_dir}")
    
    def get_lightcurve(self, source_id: int, filter_band: str = 'Ks') -> Dict[str, np.ndarray]:
        """
        Retrieve lightcurve for a given source ID.
        
        Args:
            source_id: VIRAC source identifier
            filter_band: Filter band to extract (default: 'Ks')
            
        Returns:
            Dictionary containing lightcurve data with keys:
                - mag: Magnitude values
                - magerr: Magnitude errors
                - time: Time values (MJD)
                - chi: Chi-squared values
                - ast_res_chisq: Astrometric residual chi-squared
                - filter: Filter band
                - sourceid: Source ID
                (and other VIRAC columns)
                
        Raises:
            FileNotFoundError: If FITS file for source not found
            ValueError: If no data found for specified filter
        """
        fits_path = self.lc_dir / f"{source_id}.FITS"
        
        if not fits_path.exists():
            raise FileNotFoundError(f"Lightcurve file not found: {fits_path}")
        
        try:
            with fits.open(fits_path) as hdul:
                data = hdul[1].data
                
                # Extract data for specified filter
                lightcurve = self._extract_filter_data(data, filter_band)
                
                if len(lightcurve['mag']) == 0:
                    raise ValueError(f"No data found for filter {filter_band}")
                
                logger.debug(f"Loaded lightcurve for source {source_id}: {len(lightcurve['mag'])} points")
                return lightcurve
                
        except Exception as e:
            logger.error(f"Error loading lightcurve for source {source_id}: {e}")
            raise
    
    def _extract_filter_data(self, data: np.ndarray, filter_band: str) -> Dict[str, np.ndarray]:
        """
        Extract data for a specific filter band from VIRAC FITS data.
        
        Args:
            data: FITS table data
            filter_band: Filter band to extract
            
        Returns:
            Dictionary of extracted arrays
        """
        # Find indices for specified filter
        filter_mask = data['filter'].astype(str) == filter_band
        
        # Extract relevant columns
        lightcurve = {
            'mjdobs': data['mjdobs'][filter_mask],
            'hfad_mag': data['hfad_mag'][filter_mask],
            'hfad_emag': data['hfad_emag'][filter_mask],
            'filter': data['filter'][filter_mask],
            'sourceid': data['sourceid'][filter_mask],
            'chi': data['chi'][filter_mask],
            'ast_res_chisq': data['ast_res_chisq'][filter_mask],
            'filename': data['filename'][filter_mask],
            'tile': data['tile'][filter_mask],
            'seeing': data['seeing'][filter_mask],
            'exptime': data['exptime'][filter_mask],
            'skylevel': data['skylevel'][filter_mask],
            'ellipticity': data['ellipticity'][filter_mask],
            'ambiguous_match': data['ambiguous_match'][filter_mask],
            'x': data['x'][filter_mask],
            'y': data['y'][filter_mask],
        }
        
        # Rename for convenience
        lightcurve['mag'] = lightcurve['hfad_mag']
        lightcurve['magerr'] = lightcurve['hfad_emag']
        lightcurve['time'] = lightcurve['mjdobs']
        
        return lightcurve


def load_lightcurve(
    source_id: int,
    lc_dir: str,
    filter_band: str = 'Ks'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to load lightcurve data.
    
    Args:
        source_id: VIRAC source identifier
        lc_dir: Directory containing lightcurve FITS files
        filter_band: Filter band to extract
        
    Returns:
        Tuple of (mag, magerr, time) arrays
        
    Example:
        >>> mag, magerr, time = load_lightcurve(123456, '/path/to/virac/lcs')
    """
    interface = ViracInterface(lc_dir)
    lc = interface.get_lightcurve(source_id, filter_band)
    return lc['mag'], lc['magerr'], lc['time']

