"""
File I/O Module

Handles reading and writing of catalogue data in various formats.

Author: Niall Miller
Date: 2025-10-21
"""

import pandas as pd
import numpy as np
from astropy.table import Table
from astropy.io import fits
from pathlib import Path
from typing import Union, List, Optional
import logging

logger = logging.getLogger(__name__)


def save_catalogue(
    data: Union[pd.DataFrame, Table, dict],
    output_path: str,
    format: str = 'fits',
    overwrite: bool = True
) -> None:
    """
    Save catalogue data to file.
    
    Supports FITS, CSV, and HDF5 formats.
    
    Args:
        data: Catalogue data (DataFrame, Astropy Table, or dict)
        output_path: Output file path
        format: Output format ('fits', 'csv', 'hdf5')
        overwrite: Whether to overwrite existing file
        
    Example:
        >>> save_catalogue(catalogue_df, 'primvs_catalogue.fits', format='fits')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving catalogue to {output_path} (format: {format})")
    
    # Convert to appropriate format
    if format.lower() == 'fits':
        if isinstance(data, pd.DataFrame):
            table = Table.from_pandas(data)
        elif isinstance(data, dict):
            table = Table(data)
        else:
            table = data
        
        table.write(str(output_path), format='fits', overwrite=overwrite)
        
    elif format.lower() == 'csv':
        if isinstance(data, Table):
            df = data.to_pandas()
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data
        
        df.to_csv(output_path, index=False)
        
    elif format.lower() == 'hdf5' or format.lower() == 'h5':
        if isinstance(data, Table):
            df = data.to_pandas()
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data
        
        df.to_hdf(output_path, key='catalogue', mode='w')
        
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Catalogue saved successfully ({len(data)} rows)")


def load_catalogue(
    input_path: str,
    format: Optional[str] = None
) -> pd.DataFrame:
    """
    Load catalogue data from file.
    
    Auto-detects format from extension if not specified.
    
    Args:
        input_path: Input file path
        format: File format ('fits', 'csv', 'hdf5'). If None, auto-detect.
        
    Returns:
        Catalogue data as pandas DataFrame
        
    Example:
        >>> catalogue = load_catalogue('primvs_catalogue.fits')
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Catalogue file not found: {input_path}")
    
    # Auto-detect format from extension
    if format is None:
        ext = input_path.suffix.lower()
        if ext in ['.fits', '.fit']:
            format = 'fits'
        elif ext == '.csv':
            format = 'csv'
        elif ext in ['.hdf5', '.h5']:
            format = 'hdf5'
        else:
            raise ValueError(f"Cannot auto-detect format from extension: {ext}")
    
    logger.info(f"Loading catalogue from {input_path} (format: {format})")
    
    # Load data
    if format.lower() == 'fits':
        with fits.open(input_path) as hdul:
            table = Table.read(hdul[1])
            df = table.to_pandas()
            
    elif format.lower() == 'csv':
        df = pd.read_csv(input_path)
        
    elif format.lower() in ['hdf5', 'h5']:
        df = pd.read_hdf(input_path, key='catalogue')
        
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Catalogue loaded successfully ({len(df)} rows)")
    return df


def save_lightcurve(
    mag: np.ndarray,
    magerr: np.ndarray,
    time: np.ndarray,
    phase: np.ndarray,
    output_path: str,
    metadata: Optional[dict] = None
) -> None:
    """
    Save lightcurve data to numpy file.
    
    Args:
        mag: Magnitude array
        magerr: Magnitude error array
        time: Time array
        phase: Phase array
        output_path: Output file path
        metadata: Optional metadata dictionary
        
    Example:
        >>> save_lightcurve(mag, magerr, time, phase, 'lc_12345.npy')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create structured array
    lightcurve = np.array(
        list(zip(mag, magerr, time, phase)),
        dtype=[('mag', float), ('magerr', float), ('time', float), ('phase', float)]
    )
    
    # Save with metadata if provided
    if metadata is not None:
        np.save(output_path, {'lightcurve': lightcurve, 'metadata': metadata})
    else:
        np.save(output_path, lightcurve)
    
    logger.debug(f"Saved lightcurve to {output_path}")


def load_lightcurve_file(input_path: str) -> dict:
    """
    Load lightcurve from numpy file.
    
    Args:
        input_path: Input file path
        
    Returns:
        Dictionary with lightcurve data and metadata
        
    Example:
        >>> data = load_lightcurve_file('lc_12345.npy')
        >>> mag = data['lightcurve']['mag']
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Lightcurve file not found: {input_path}")
    
    data = np.load(input_path, allow_pickle=True)
    
    if isinstance(data, np.ndarray) and data.dtype.names:
        # Simple structured array
        return {'lightcurve': data, 'metadata': None}
    else:
        # Dictionary with metadata
        return data.item()

