"""
Main Pipeline Module

Orchestrates the complete PRIMVS catalogue construction pipeline.

This module ties together all components (data access, preprocessing,
feature extraction, periodogram analysis, FAP calculation, and aggregation)
into a cohesive pipeline that can process VIRAC lightcurves to produce
the PRIMVS catalogue.

Methodology follows Miller et al. (2026):
  1. Load and clean lightcurve (quality cuts, sigma clip, detrend)
  2. Calculate statistical features (variability indices, moments)
  3. Run periodograms: LS, PDM, CE, GP
  4. Extract peaks, check aliases
  5. Calculate FAP on best candidate periods
  6. Output comprehensive catalogue

Author: Niall Miller (refactored)
Date: 2025-10-21
Updated: 2026-02-16 — Integrated periodogram analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import logging

logger = logging.getLogger(__name__)


from .config import load_config, get_data_paths, get_processing_params
from .data_access import ViracInterface, save_catalogue
from .preprocessing import QualityFilter
from .features import FeatureCalculator
from .fap import NeuralNetworkFAP
from .utils import parallel_process
from .utils.logging_config import get_logger

# Periodogram methods from stochistats
try:
    from stochistats import (
        LS, PDM, CE, GP,
        extract_peaks, check_alias, exclude_alias_regions,
        make_frequency_grid,
    )
    PERIODOGRAMS_AVAILABLE = True
except ImportError:
    PERIODOGRAMS_AVAILABLE = False

logger = get_logger(__name__)


# Default periodogram parameters
DEFAULT_N_FREQS = 100_000
DEFAULT_F_STOP = 10.0
DEFAULT_N_PEAKS = 2
DEFAULT_FAP_THRESHOLD = 0.2





def generate_primvs_id(sourceid: Union[int, str],
                       true_period: float,
                       best_fap: float) -> int:
    """
    Generate a PRIMVS unique ID from source ID, period, and FAP.

    Replicates file_aggrigator.py line 189:
        sourceid_str[:-2] + period*1000 (zero-padded 3) + fap*100 (zero-padded 2)

    Parameters
    ----------
    sourceid : int or str
        VIRAC source identifier.
    true_period : float
        Best-fit period in days.
    best_fap : float
        Best false alarm probability (0-1).

    Returns
    -------
    int
        PRIMVS unique identifier.
    """
    sourceid_str = str(int(sourceid))
    sourceid_trimmed = sourceid_str[:-2]
    period_code = '{:0>3d}'.format(int(true_period * 1000))
    fap_code = '{:0>2d}'.format(int(best_fap * 100))
    return int(sourceid_trimmed + period_code + fap_code)


def generate_period_id(true_period: float) -> int:
    """
    Generate a period ID encoding the period value.

    Replicates file_aggrigator.py line 185:
        int('{:0>8d}'.format(int(period * 100000)))

    Parameters
    ----------
    true_period : float
        Best-fit period in days.

    Returns
    -------
    int
        Period identifier.
    """
    return int('{:0>8d}'.format(int(true_period * 100000)))


def add_primvs_ids_to_result(result: Dict) -> Dict:
    """
    Add primvs_id and period_id to a single pipeline result dict.

    Call this at the end of Pipeline.process_source() so IDs are
    generated in-situ rather than as a post-processing step.

    Parameters
    ----------
    result : dict
        Pipeline result containing 'sourceid', 'true_period', 'best_fap'.

    Returns
    -------
    dict
        Same dict with 'primvs_id' and 'period_id' added.
    """
    sourceid = result.get('sourceid')
    true_period = float(result.get('true_period', 0.0) or 0.0)
    best_fap = float(result.get('best_fap', 1.0) or 1.0)

    if sourceid is None:
        logger.warning("Cannot generate PRIMVS ID: sourceid is None")
        return result

    result['primvs_id'] = generate_primvs_id(sourceid, true_period, best_fap)
    result['period_id'] = generate_period_id(true_period)
    return result


def add_primvs_ids_to_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add primvs_id and period_id columns to a catalogue DataFrame.

    Replicates the full file_aggrigator.py logic (lines 185-197):
    generates the IDs, casts types, and reorders columns so
    primvs_id and sourceid come first.

    Parameters
    ----------
    df : pd.DataFrame
        Catalogue with 'sourceid', 'true_period', 'best_fap' columns.

    Returns
    -------
    pd.DataFrame
        Catalogue with 'primvs_id', 'sourceid', 'period_id' as first columns.
    """
    df = df.copy()

    df['period_id'] = df['true_period'].apply(
        lambda p: int('{:0>8d}'.format(int(float(p) * 100000)))
    )

    df['primvs_id'] = (
        df['sourceid'].astype(str).str[:-2]
        + df['true_period'].apply(
            lambda p: '{:0>3d}'.format(int(float(p) * 1000))
        ).astype(str)
        + df['best_fap'].apply(
            lambda f: '{:0>2d}'.format(int(float(f) * 100))
        ).astype(str)
    )

    df['primvs_id'] = df['primvs_id'].astype(int)
    df['sourceid'] = df['sourceid'].astype(int)
    df['period_id'] = df['period_id'].astype(int)

    # Reorder: primvs_id, sourceid, period_id first
    id_cols = ['primvs_id', 'sourceid', 'period_id']
    other_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + other_cols]

    logger.info(f"Generated PRIMVS IDs for {len(df)} rows")
    return df



class Pipeline:
    """
    Main PRIMVS pipeline orchestrator.

    Coordinates all stages of catalogue construction from raw lightcurves
    to final catalogue output, including full periodogram analysis.

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
        if config is None:
            config = load_config()

        self.config = config

        # Get configuration sections
        self.paths = get_data_paths(config)
        self.proc_params = get_processing_params(config)
        self.quality_config = config.get('quality_filters', {})
        self.fap_config = config.get('fap', {})
        self.feature_config = config.get('features', {})

        # Periodogram configuration
        self.periodogram_methods = self.feature_config.get(
            'periodogram_methods', ['lomb_scargle', 'pdm', 'conditional_entropy', 'gaussian_process']
        )
        self.n_freqs = self.proc_params.get('n_freqs', DEFAULT_N_FREQS)
        self.fap_threshold = self.fap_config.get('threshold', DEFAULT_FAP_THRESHOLD)

        # Initialize components
        logger.info("Initializing pipeline components...")

        # VIRAC interface
        self.virac = ViracInterface(
            lc_dir=str(self.paths['virac_lightcurves']),
            meta_dir=str(self.paths.get('virac_meta', ''))
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
        fap_model_path = self._resolve_fap_model_path()
        if fap_model_path is not None and fap_model_path.exists():
            try:
                self.fap_calc = NeuralNetworkFAP(
                    model_path=str(fap_model_path),
                    n_points=self.fap_config.get('n_points', 200),
                    knn_neighbors=self.fap_config.get('knn_neighbors', 10)
                )
                logger.info(f"FAP calculator initialized from {fap_model_path}")
            except Exception as e:
                logger.warning(f"Could not initialize FAP calculator: {e}")
                self.fap_calc = None
        else:
            logger.warning(f"FAP model not found at {fap_model_path}")
            self.fap_calc = None

        if not PERIODOGRAMS_AVAILABLE:
            logger.error("stochistats periodogram methods not available! Period finding will be skipped.")

        logger.info("Pipeline initialized successfully")

    def _resolve_fap_model_path(self) -> Optional[Path]:
        """Resolve FAP model path from config, trying absolute then relative."""
        # Try fap section first
        fap_path_str = self.fap_config.get('model_path', '')
        if fap_path_str:
            p = Path(fap_path_str)
            if p.is_absolute() and p.exists():
                return p

        # Try data section
        fap_rel = self.config['data'].get('fap_model_path', '')
        if fap_rel:
            # Try relative to models_dir
            p = self.paths.get('models', Path('./models')) / fap_rel
            if p.exists():
                return p
            # Try as absolute
            p = Path(fap_rel)
            if p.exists():
                return p

        return self.paths.get('models', Path('./models')) / 'fap_nn' / 'final_12l_dp_all'

    # ------------------------------------------------------------------
    # Periodogram analysis
    # ------------------------------------------------------------------
    def _run_periodograms(
        self,
        mag: np.ndarray,
        magerr: np.ndarray,
        time: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run all configured periodogram methods and extract peaks.

        Follows the PRIMVS methodology:
        - LS and PDM as primary methods
        - CE as complementary method
        - GP for quasi-periodic fitting

        Args:
            mag: Magnitude array (cleaned)
            magerr: Magnitude error array
            time: Time array (MJD)

        Returns:
            Dictionary with period results from all methods
        """
        results = {}
        n_freqs = self.n_freqs

        # --- Lomb-Scargle ---
        if 'lomb_scargle' in self.periodogram_methods:
            try:
                ls_freqs, ls_power = LS(mag, magerr, time, n_freqs=n_freqs)
                ls_freqs_clean, ls_power_clean = exclude_alias_regions(ls_freqs, ls_power)
                ls_peaks = extract_peaks(ls_freqs_clean, ls_power_clean, n_peaks=DEFAULT_N_PEAKS, minimize=False)

                if len(ls_peaks) >= 1:
                    results['ls_period1'] = ls_peaks[0]['period']
                    results['ls_power1'] = ls_peaks[0]['power']
                if len(ls_peaks) >= 2:
                    results['ls_period2'] = ls_peaks[1]['period']
                    results['ls_power2'] = ls_peaks[1]['power']

                logger.debug(f"LS: top period = {results.get('ls_period1', 'N/A')}")
            except Exception as e:
                logger.warning(f"LS periodogram failed: {e}")

        # --- Phase Dispersion Minimization ---
        if 'pdm' in self.periodogram_methods:
            try:
                pdm_freqs, pdm_theta = PDM(mag, magerr, time, n_freqs=n_freqs)
                pdm_freqs_clean, pdm_theta_clean = exclude_alias_regions(pdm_freqs, pdm_theta)
                pdm_peaks = extract_peaks(pdm_freqs_clean, pdm_theta_clean, n_peaks=DEFAULT_N_PEAKS, minimize=True)

                if len(pdm_peaks) >= 1:
                    results['pdm_period1'] = pdm_peaks[0]['period']
                    results['pdm_theta1'] = pdm_peaks[0]['power']
                if len(pdm_peaks) >= 2:
                    results['pdm_period2'] = pdm_peaks[1]['period']
                    results['pdm_theta2'] = pdm_peaks[1]['power']

                logger.debug(f"PDM: top period = {results.get('pdm_period1', 'N/A')}")
            except Exception as e:
                logger.warning(f"PDM periodogram failed: {e}")

        # --- Conditional Entropy ---
        if 'conditional_entropy' in self.periodogram_methods:
            try:
                ce_freqs, ce_entropy = CE(mag, magerr, time, n_freqs=n_freqs)
                ce_freqs_clean, ce_entropy_clean = exclude_alias_regions(ce_freqs, ce_entropy)
                ce_peaks = extract_peaks(ce_freqs_clean, ce_entropy_clean, n_peaks=DEFAULT_N_PEAKS, minimize=True)

                if len(ce_peaks) >= 1:
                    results['ce_period1'] = ce_peaks[0]['period']
                    results['ce_entropy1'] = ce_peaks[0]['power']
                if len(ce_peaks) >= 2:
                    results['ce_period2'] = ce_peaks[1]['period']
                    results['ce_entropy2'] = ce_peaks[1]['power']

                logger.debug(f"CE: top period = {results.get('ce_period1', 'N/A')}")
            except Exception as e:
                logger.warning(f"CE periodogram failed: {e}")

        # --- Gaussian Process ---
        if 'gaussian_process' in self.periodogram_methods:
            try:
                gp_result = GP(mag, magerr, time)
                if gp_result and not np.isnan(gp_result.get('period', np.nan)):
                    results['gp_period'] = gp_result['period']
                    results['gp_log_likelihood'] = gp_result.get('log_likelihood', np.nan)
                    results['gp_b'] = gp_result.get('b', np.nan)
                    results['gp_c'] = gp_result.get('c', np.nan)

                logger.debug(f"GP: period = {results.get('gp_period', 'N/A')}")
            except Exception as e:
                logger.warning(f"GP period finding failed: {e}")

        # --- Determine best overall period ---
        results['true_period'] = self._select_best_period(results)

        return results

    def _select_best_period(self, period_results: Dict[str, Any]) -> float:
        """
        Select the best period from all methods.

        Priority: use the period that appears most consistently across methods.
        If no clear consensus, prefer LS period.

        Args:
            period_results: Dictionary with period results from all methods

        Returns:
            Best period estimate (or NaN if none found)
        """
        candidates = []

        for key in ['ls_period1', 'pdm_period1', 'ce_period1', 'gp_period']:
            p = period_results.get(key, np.nan)
            if not np.isnan(p) and p > 0:
                candidates.append(p)

        if not candidates:
            return np.nan

        # Check for consensus: if multiple methods agree within 1%, use that
        for i, p1 in enumerate(candidates):
            agreeing = [p for p in candidates if abs(p - p1) / p1 < 0.01]
            if len(agreeing) >= 2:
                return float(np.median(agreeing))

        # No consensus — prefer LS if available, else first available
        return period_results.get('ls_period1',
               period_results.get('pdm_period1',
               period_results.get('ce_period1',
               period_results.get('gp_period', np.nan))))

    # ------------------------------------------------------------------
    # Source processing
    # ------------------------------------------------------------------
    def process_source(
        self,
        source_id: int,
        period: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single source through the full PRIMVS pipeline.

        Steps:
          1. Load lightcurve from VIRAC CSV
          2. Apply quality filters
          3. Calculate statistical features (variability indices, moments)
          4. Run periodograms (LS, PDM, CE, GP) and extract peaks
          5. Calculate FAP on the best period
          6. Return combined feature dictionary

        Args:
            source_id: VIRAC source identifier
            period: Override period (if provided, skip period-finding).
                    Mainly for testing/validation.

        Returns:
            Dictionary of all features, or None if processing failed
        """
        try:
            # 1. Load lightcurve
            logger.debug(f"Processing source {source_id}")
            lc = self.virac.get_lightcurve(source_id, filter_band='Ks')

            # 2. Apply quality filters
            filtered_lc = self.quality_filter.apply(lc)

            # Check minimum observations
            min_obs = self.proc_params.get('min_observations', 40)
            if len(filtered_lc['mag']) < min_obs:
                logger.debug(f"Source {source_id}: only {len(filtered_lc['mag'])} obs (< {min_obs})")
                return None

            # Extract arrays
            mag = filtered_lc['mag']
            magerr = filtered_lc['magerr']
            time = filtered_lc['time']

            # 3. Calculate statistical features
            features = self.feature_calc.calculate_all(mag, magerr, time)
            features['sourceid'] = source_id

            # 4. Run periodograms (or use provided period)
            if period is not None and not np.isnan(period):
                # Period provided externally — skip periodogram computation
                features['true_period'] = period
                logger.debug(f"Source {source_id}: using provided period {period:.6f}")
            elif PERIODOGRAMS_AVAILABLE:
                # Run full periodogram analysis
                period_results = self._run_periodograms(mag, magerr, time)
                features.update(period_results)
                period = features.get('true_period', np.nan)
            else:
                features['true_period'] = np.nan
                period = np.nan

            # 5. Calculate FAP if we have a period and FAP calculator
            if self.fap_calc is not None and period is not None and not np.isnan(period):
                try:
                    fap = self.fap_calc.calculate(period, mag, time)
                    features['best_fap'] = fap
                except Exception as e:
                    logger.warning(f"FAP calculation failed for source {source_id}: {e}")
                    features['best_fap'] = np.nan
            else:
                features['best_fap'] = np.nan

            logger.debug(f"Source {source_id}: period={features.get('true_period', 'N/A')}, "
                         f"FAP={features.get('best_fap', 'N/A')}")


            add_primvs_ids_to_result(period_results)

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

        if periods is None:
            periods = [None] * len(source_ids)

        if n_processes is None:
            n_processes = self.proc_params.get('n_processes', -1)

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

        valid_results = [r for r in results if r is not None]

        logger.info(f"Successfully processed {len(valid_results)}/{len(source_ids)} sources")

        if len(valid_results) > 0:
            return pd.DataFrame(valid_results)
        else:
            logger.warning("No sources were successfully processed")
            return pd.DataFrame()

    def save_catalogue(
        self,
        catalogue: pd.DataFrame,
        output_name: str = "primvs_catalogue",
        formats: Optional[List[str]] = None
    ) -> None:
        """Save catalogue to disk."""
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

        Args:
            source_ids: List of source IDs to process
            periods: List of periods (optional, None = compute from scratch)
            output_name: Base name for output catalogue

        Returns:
            Catalogue DataFrame
        """
        logger.info("Starting PRIMVS pipeline")

        catalogue = self.process_sources(source_ids, periods)

        if len(catalogue) > 0:
            self.save_catalogue(catalogue, output_name)
        else:
            logger.warning("No sources to save - catalogue is empty")

        logger.info("Pipeline complete")
        return catalogue
