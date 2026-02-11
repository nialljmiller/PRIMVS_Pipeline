"""
Main Pipeline Module

Orchestrates the complete PRIMVS catalogue construction pipeline.

This module ties together all components (data access, preprocessing,
feature extraction, FAP calculation, and aggregation) into a cohesive
pipeline that can process VIRAC lightcurves to produce the PRIMVS catalogue.

Author: Niall Miller (refactored)
Date: 2025-10-21
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

from .config import load_config, get_data_paths, get_processing_params
from .data_access import ViracInterface, save_catalogue
from .preprocessing import QualityFilter
from .features import FeatureCalculator
from .fap import NeuralNetworkFAP
from .utils import parallel_process
from .utils.logging_config import get_logger


logger = get_logger(__name__)


class Pipeline:
    """
    Main PRIMVS pipeline orchestrator.
    
    Coordinates all stages of catalogue construction from raw lightcurves
    to final catalogue output.
    
    Attributes:
        config: Pipeline configuration dictionary
        virac: VIRAC data interface
        quality_filter: Quality filter instance
        feature_calc: Feature calculator instance
        fap_calc: FAP calculator instance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration dictionary. If None, loads from default location.
        """
        # Load configuration
        if config is None:
            config = load_config()
        
        self.config = config
        
        # Get configuration sections
        self.paths = get_data_paths(config)
        self.proc_params = get_processing_params(config)
        self.quality_config = config.get('quality_filters', {})
        self.fap_config = config.get('fap', {})
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        # VIRAC interface
        self.virac = ViracInterface(
            lc_dir=str(self.paths['virac_lightcurves']),
            meta_dir=str(self.paths['virac_meta'])
        )
        
        # Quality filter
        self.quality_filter = QualityFilter(
            max_chi=self.quality_config.get('max_chi', 10.0),
            max_ast_res_chisq=self.quality_config.get('max_ast_res_chisq', 20.0),
            max_magerr_sigma=self.quality_config.get('max_magerr_sigma', 4.0),
            require_positive_mag=self.quality_config.get('require_positive_mag', True),
            require_positive_magerr=self.quality_config.get('require_positive_magerr', True)
        )
        
        # Feature calculator
        self.feature_calc = FeatureCalculator()
        
        # FAP calculator
        fap_model_path = self.paths['models'] / self.config['data'].get('fap_model_path', '')
        if fap_model_path.exists():
            try:
                self.fap_calc = NeuralNetworkFAP(
                    model_path=str(fap_model_path),
                    n_points=self.fap_config.get('n_points', 200),
                    knn_neighbors=self.fap_config.get('knn_neighbors', 10)
                )
                logger.info("FAP calculator initialized")
            except Exception as e:
                logger.warning(f"Could not initialize FAP calculator: {e}")
                self.fap_calc = None
        else:
            logger.warning(f"FAP model not found at {fap_model_path}")
            self.fap_calc = None
        
        logger.info("Pipeline initialized successfully")
    
    def process_source(
        self,
        source_id: int,
        period: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single source.
        
        Args:
            source_id: VIRAC source identifier
            period: Known period (if available). If None, period-finding is skipped.
            
        Returns:
            Dictionary of features, or None if processing failed
        """
        try:
            # Load lightcurve
            logger.debug(f"Processing source {source_id}")
            lc = self.virac.get_lightcurve(source_id, filter_band='Ks')
            
            # Apply quality filters
            filtered_lc = self.quality_filter.apply(lc)
            
            # Check minimum observations
            min_obs = self.proc_params.get('min_observations', 40)
            if len(filtered_lc['mag']) < min_obs:
                logger.debug(f"Source {source_id} has only {len(filtered_lc['mag'])} observations (< {min_obs})")
                return None
            
            # Extract arrays
            mag = filtered_lc['mag']
            magerr = filtered_lc['magerr']
            time = filtered_lc['time']
            
            # Calculate statistical features
            features = self.feature_calc.calculate_all(mag, magerr, time)
            
            # Add source ID
            features['sourceid'] = source_id
            
            # Calculate FAP if period provided and FAP calculator available
            if period is not None and self.fap_calc is not None:
                try:
                    fap = self.fap_calc.calculate(period, mag, time)
                    features['best_fap'] = fap
                    features['true_period'] = period
                except Exception as e:
                    logger.warning(f"FAP calculation failed for source {source_id}: {e}")
                    features['best_fap'] = np.nan
                    features['true_period'] = period
            
            logger.debug(f"Successfully processed source {source_id}")
            return features
            
        except FileNotFoundError:
            logger.debug(f"Lightcurve file not found for source {source_id}")
            return None
        except Exception as e:
            logger.error(f"Error processing source {source_id}: {e}")
            return None
    
    def process_sources(
        self,
        source_ids: List[int],
        periods: Optional[List[float]] = None,
        n_processes: Optional[int] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Process multiple sources in parallel.
        
        Args:
            source_ids: List of source IDs to process
            periods: List of periods (same length as source_ids), or None
            n_processes: Number of parallel processes
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame containing features for all sources
        """
        logger.info(f"Processing {len(source_ids)} sources")
        
        # Prepare arguments
        if periods is None:
            periods = [None] * len(source_ids)
        
        if n_processes is None:
            n_processes = self.proc_params.get('n_processes', -1)
        
        # Process in parallel
        def process_wrapper(args):
            source_id, period = args
            return self.process_source(source_id, period)
        
        args_list = list(zip(source_ids, periods))
        
        results = parallel_process(
            process_wrapper,
            args_list,
            n_processes=n_processes,
            show_progress=show_progress,
            desc="Processing sources"
        )
        
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        
        logger.info(f"Successfully processed {len(valid_results)}/{len(source_ids)} sources")
        
        # Convert to DataFrame
        if len(valid_results) > 0:
            df = pd.DataFrame(valid_results)
            return df
        else:
            logger.warning("No sources were successfully processed")
            return pd.DataFrame()
    
    def save_catalogue(
        self,
        catalogue: pd.DataFrame,
        output_name: str = "primvs_catalogue",
        formats: Optional[List[str]] = None
    ) -> None:
        """
        Save catalogue to disk.
        
        Args:
            catalogue: Catalogue DataFrame
            output_name: Base name for output files (without extension)
            formats: List of formats to save ('fits', 'csv', 'hdf5')
        """
        if formats is None:
            formats = self.config.get('aggregation', {}).get('output_formats', ['fits', 'csv'])
        
        output_dir = self.paths['output']
        
        for fmt in formats:
            output_path = output_dir / f"{output_name}.{fmt}"
            logger.info(f"Saving catalogue to {output_path}")
            save_catalogue(catalogue, str(output_path), format=fmt)
        
        logger.info(f"Catalogue saved ({len(catalogue)} sources)")
    
    def run(
        self,
        source_ids: List[int],
        periods: Optional[List[float]] = None,
        output_name: str = "primvs_catalogue"
    ) -> pd.DataFrame:
        """
        Run complete pipeline.
        
        Convenience method that processes sources and saves the catalogue.
        
        Args:
            source_ids: List of source IDs to process
            periods: List of periods (optional)
            output_name: Base name for output catalogue
            
        Returns:
            Catalogue DataFrame
        """
        logger.info("Starting PRIMVS pipeline")
        
        # Process sources
        catalogue = self.process_sources(source_ids, periods)
        
        # Save catalogue
        if len(catalogue) > 0:
            self.save_catalogue(catalogue, output_name)
        else:
            logger.warning("No sources to save - catalogue is empty")
        
        logger.info("Pipeline complete")
        
        return catalogue

