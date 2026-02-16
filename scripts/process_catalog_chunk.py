"""
Process a chunk of the PRIMVS catalog with row-by-row CSV append.

Each source is written to the output CSV immediately upon completion,
matching the original PRIMVS pipeline behavior (PRIMVS_file.py).
This ensures partial results are saved even if the job is killed.

Output columns match PRIMVS_FULL.fits exactly (102 columns):
    uniqueid, sourceid, mag_n, ..., Cody_Q_gp

The uniqueid (PRIMVS ID) is generated in-situ using the same formula
as file_aggrigator.py:
    str(sourceid)[:-2] + period*1000 (3-digit) + fap*100 (2-digit)

At completion, the CSV is also written as a FITS file.
"""
import argparse
import csv
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.table import Table

from primvs_pipeline.config import load_config, get_data_paths, get_processing_params
from primvs_pipeline.data_access import ViracInterface
from primvs_pipeline.preprocessing import QualityFilter
from primvs_pipeline.features import FeatureCalculator

try:
    from stochistats import (
        LS, PDM, CE, GP,
        extract_peaks, check_alias, exclude_alias_regions,
        phaser,
    )
    PERIODOGRAMS_AVAILABLE = True
except ImportError:
    PERIODOGRAMS_AVAILABLE = False

try:
    from stochistats import cody_Q
    CODY_Q_AVAILABLE = True
except ImportError:
    CODY_Q_AVAILABLE = False

try:
    from primvs_pipeline.fap import NeuralNetworkFAP
    FAP_AVAILABLE = True
except ImportError:
    FAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# PRIMVS ID generation (from file_aggrigator.py line 189)
# ---------------------------------------------------------------------------

def generate_primvs_id(sourceid, true_period, best_fap):
    """
    Generate PRIMVS unique ID.

    Formula (file_aggrigator.py):
        str(sourceid)[:-2] + '{:0>3d}'.format(int(period * 1000))
                            + '{:0>2d}'.format(int(fap * 100))
    """
    sourceid_str = str(int(sourceid))
    sourceid_trimmed = sourceid_str[:-2]
    period_code = '{:0>3d}'.format(int(float(true_period) * 1000))
    fap_code = '{:0>2d}'.format(int(float(best_fap) * 100))
    return int(sourceid_trimmed + period_code + fap_code)


# ---------------------------------------------------------------------------
# Column order — matches PRIMVS_FULL.fits exactly (102 columns)
# ---------------------------------------------------------------------------
COL_NAMES = [
    'uniqueid', 'sourceid',
    'mag_n', 'mag_avg', 'magerr_avg',
    'Cody_M', 'stet_k', 'eta', 'eta_e', 'med_BRP',
    'range_cum_sum', 'max_slope', 'MAD', 'mean_var',
    'percent_amp', 'true_amplitude', 'roms', 'p_to_p_var',
    'lag_auto', 'AD', 'std_nxs',
    'weight_mean', 'weight_std', 'weight_skew', 'weight_kurt',
    'mean', 'std', 'skew', 'kurt',
    'time_range', 'true_period', 'true_class', 'best_fap', 'best_method', 'trans_flag',
    # LS
    'ls_p', 'ls_y_y_0', 'ls_peak_width_0',
    'ls_period1', 'ls_y_y_1', 'ls_peak_width_1',
    'ls_period2', 'ls_y_y_2', 'ls_peak_width_2',
    'ls_q001', 'ls_q01', 'ls_q1', 'ls_q25', 'ls_q50', 'ls_q75', 'ls_q99', 'ls_q999', 'ls_q9999',
    'ls_fap', 'ls_bal_fap', 'Cody_Q_ls',
    # PDM
    'pdm_p', 'pdm_y_y_0', 'pdm_peak_width_0',
    'pdm_period1', 'pdm_y_y_1', 'pdm_peak_width_1',
    'pdm_period2', 'pdm_y_y_2', 'pdm_peak_width_2',
    'pdm_q001', 'pdm_q01', 'pdm_q1', 'pdm_q25', 'pdm_q50', 'pdm_q75', 'pdm_q99', 'pdm_q999', 'pdm_q9999',
    'pdm_fap', 'Cody_Q_pdm',
    # CE
    'ce_p', 'ce_y_y_0', 'ce_peak_width_0',
    'ce_period1', 'ce_y_y_1', 'ce_peak_width_1',
    'ce_period2', 'ce_y_y_2', 'ce_peak_width_2',
    'ce_q001', 'ce_q01', 'ce_q1', 'ce_q25', 'ce_q50', 'ce_q75', 'ce_q99', 'ce_q999', 'ce_q9999',
    'ce_fap', 'Cody_Q_ce',
    # GP
    'gp_lnlike', 'gp_b', 'gp_c', 'gp_p', 'gp_fap', 'Cody_Q_gp',
]

assert len(COL_NAMES) == 102, f"Expected 102 columns, got {len(COL_NAMES)}"

QUANTILES = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.99, 0.999, 0.9999]
N_FREQS = 100_000
N_PEAKS = 3


# ---------------------------------------------------------------------------
# Periodogram runners
# ---------------------------------------------------------------------------

def _quantiles(power):
    return [float(np.nanquantile(power, q)) for q in QUANTILES]


def run_ls(mag, magerr, time):
    r = {}
    try:
        freqs, power = LS(mag, magerr, time, n_freqs=N_FREQS)
        freqs_c, power_c = exclude_alias_regions(freqs, power)
        peaks = extract_peaks(freqs_c, power_c, n_peaks=N_PEAKS, minimize=False)
        qs = _quantiles(power)
        for i, pk in enumerate(peaks[:3]):
            key_p = 'ls_p' if i == 0 else f'ls_period{i}'
            r[key_p]                 = pk['period']
            r[f'ls_y_y_{i}']        = pk['power']
            r[f'ls_peak_width_{i}'] = pk['width']
        for i, qn in enumerate(['ls_q001','ls_q01','ls_q1','ls_q25','ls_q50','ls_q75','ls_q99','ls_q999','ls_q9999']):
            r[qn] = qs[i]
    except Exception as e:
        print(f"  LS failed: {e}", file=sys.stderr)
    return r


def run_pdm(mag, magerr, time):
    r = {}
    try:
        freqs, theta = PDM(mag, magerr, time, n_freqs=N_FREQS)
        freqs_c, theta_c = exclude_alias_regions(freqs, theta)
        peaks = extract_peaks(freqs_c, theta_c, n_peaks=N_PEAKS, minimize=True)
        qs = _quantiles(theta)
        for i, pk in enumerate(peaks[:3]):
            key_p = 'pdm_p' if i == 0 else f'pdm_period{i}'
            r[key_p]                  = pk['period']
            r[f'pdm_y_y_{i}']        = pk['power']
            r[f'pdm_peak_width_{i}'] = pk['width']
        for i, qn in enumerate(['pdm_q001','pdm_q01','pdm_q1','pdm_q25','pdm_q50','pdm_q75','pdm_q99','pdm_q999','pdm_q9999']):
            r[qn] = qs[i]
    except Exception as e:
        print(f"  PDM failed: {e}", file=sys.stderr)
    return r


def run_ce(mag, magerr, time):
    r = {}
    try:
        freqs, entropy = CE(mag, magerr, time, n_freqs=N_FREQS)
        freqs_c, entropy_c = exclude_alias_regions(freqs, entropy)
        peaks = extract_peaks(freqs_c, entropy_c, n_peaks=N_PEAKS, minimize=True)
        qs = _quantiles(entropy)
        for i, pk in enumerate(peaks[:3]):
            key_p = 'ce_p' if i == 0 else f'ce_period{i}'
            r[key_p]                 = pk['period']
            r[f'ce_y_y_{i}']        = pk['power']
            r[f'ce_peak_width_{i}'] = pk['width']
        for i, qn in enumerate(['ce_q001','ce_q01','ce_q1','ce_q25','ce_q50','ce_q75','ce_q99','ce_q999','ce_q9999']):
            r[qn] = qs[i]
    except Exception as e:
        print(f"  CE failed: {e}", file=sys.stderr)
    return r


def run_gp(mag, magerr, time):
    r = {}
    try:
        gp = GP(mag, magerr, time)
        r['gp_lnlike'] = gp.get('log_likelihood', np.nan)
        r['gp_b']      = gp.get('b', np.nan)
        r['gp_c']      = gp.get('c', np.nan)
        r['gp_p']      = gp.get('period', np.nan)
    except Exception as e:
        print(f"  GP failed: {e}", file=sys.stderr)
    return r


# ---------------------------------------------------------------------------
# FAP + best period selection
# ---------------------------------------------------------------------------

def compute_fap_and_select_best(row, fap_calc, mag, time):
    """
    Compute per-method FAP and Cody_Q, pick method with lowest FAP.
    """
    method_period_keys = {
        'ls':  'ls_p',
        'pdm': 'pdm_p',
        'ce':  'ce_p',
        'gp':  'gp_p',
    }

    best_fap = np.nan
    best_period = np.nan
    best_method = ''

    for method, pkey in method_period_keys.items():
        period = row.get(pkey, np.nan)
        if np.isnan(period) or period <= 0:
            row[f'{method}_fap'] = np.nan
            row[f'Cody_Q_{method}'] = np.nan
            continue

        fap = np.nan
        if fap_calc is not None:
            try:
                fap = fap_calc.calculate(period, mag, time)
            except Exception:
                pass
        row[f'{method}_fap'] = fap

        if CODY_Q_AVAILABLE:
            try:
                phase = phaser(time, period)
                row[f'Cody_Q_{method}'] = cody_Q(mag, phase)
            except Exception:
                row[f'Cody_Q_{method}'] = np.nan
        else:
            row[f'Cody_Q_{method}'] = np.nan

        if not np.isnan(fap) and (np.isnan(best_fap) or fap < best_fap):
            best_fap = fap
            best_period = period
            best_method = method

    # Fallback if no FAP available
    if np.isnan(best_period):
        for pkey in ['ls_p', 'pdm_p', 'ce_p', 'gp_p']:
            p = row.get(pkey, np.nan)
            if not np.isnan(p) and p > 0:
                best_period = p
                best_method = pkey.split('_')[0]
                break

    row['ls_bal_fap']   = row.get('ls_fap', np.nan)
    row['true_period']  = best_period
    row['best_fap']     = best_fap
    row['best_method']  = best_method
    row['true_class']   = ''
    row['trans_flag']    = 0.0

    return row


# ---------------------------------------------------------------------------
# Single-source processing
# ---------------------------------------------------------------------------

def process_single_source(source_id, virac, quality_filter, feature_calc, fap_calc, min_obs=40):
    """
    Process one source through the full pipeline.
    Returns a dict keyed by COL_NAMES, or None on failure.
    """
    row = {c: np.nan for c in COL_NAMES}
    row['sourceid'] = int(source_id)
    row['uniqueid'] = 0              # placeholder, set after FAP
    row['true_class'] = ''
    row['best_method'] = ''
    row['trans_flag'] = 0.0

    try:
        # 1. Load lightcurve
        lc = virac.get_lightcurve(source_id, filter_band='Ks')

        # 2. Quality filter
        flc = quality_filter.apply(lc)
        mag    = flc['mag']
        magerr = flc['magerr']
        time   = flc['time']

        if len(mag) < min_obs:
            return None

        # 3. Statistical features
        feat = feature_calc.calculate_all(mag, magerr, time)
        for key in ['mag_n','mag_avg','magerr_avg',
                     'Cody_M','stet_k','eta','eta_e','med_BRP',
                     'range_cum_sum','max_slope','MAD','mean_var',
                     'percent_amp','true_amplitude','roms','p_to_p_var',
                     'lag_auto','AD','std_nxs',
                     'weight_mean','weight_std','weight_skew','weight_kurt',
                     'mean','std','skew','kurt','time_range']:
            row[key] = feat.get(key, np.nan)

        # 4. Periodograms
        if PERIODOGRAMS_AVAILABLE:
            row.update(run_ls(mag, magerr, time))
            row.update(run_pdm(mag, magerr, time))
            row.update(run_ce(mag, magerr, time))
            row.update(run_gp(mag, magerr, time))

        # 5. FAP + best period
        row = compute_fap_and_select_best(row, fap_calc, mag, time)

        # 6. Generate PRIMVS ID (uniqueid)
        true_period = row.get('true_period', 0.0)
        best_fap    = row.get('best_fap', 1.0)
        if np.isnan(true_period):
            true_period = 0.0
        if np.isnan(best_fap):
            best_fap = 1.0
        row['uniqueid'] = generate_primvs_id(source_id, true_period, best_fap)

        return row

    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"  Error processing {source_id}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# CSV to FITS conversion
# ---------------------------------------------------------------------------

def csv_to_fits(csv_path, fits_path):
    """
    Read the completed CSV and write it as a FITS binary table,
    matching PRIMVS_FULL.fits format (int64 for IDs, float64 for numerics,
    64A for strings).
    """
    df = pd.read_csv(csv_path)

    # Type casting to match PRIMVS_FULL.fits
    int_cols = ['uniqueid', 'sourceid']
    str_cols = ['true_class', 'best_method']

    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype(np.int64)

    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].fillna('').astype(str)

    # Everything else is float64
    for c in df.columns:
        if c not in int_cols and c not in str_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float64)

    # Ensure column order matches COL_NAMES
    for c in COL_NAMES:
        if c not in df.columns:
            if c in str_cols:
                df[c] = ''
            elif c in int_cols:
                df[c] = 0
            else:
                df[c] = np.nan
    df = df[COL_NAMES]

    t = Table.from_pandas(df)
    t.write(fits_path, overwrite=True)
    print(f"FITS written: {fits_path} ({len(t)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Process a chunk of the PRIMVS catalog")
    parser.add_argument("--fits", type=str, required=True,
                        help="Path to the input catalog FITS file (with sourceid column)")
    parser.add_argument("--start", type=int, default=0,
                        help="Starting index in the FITS table")
    parser.add_argument("--count", type=int, default=1000,
                        help="Number of sources to process")
    parser.add_argument("--output", type=str, required=True,
                        help="Output name (writes output/<name>.csv and output/<name>.fits)")
    parser.add_argument("--config", type=str, default="../config/pipeline_config.yaml")
    args = parser.parse_args()

    # --- Config ---
    config = load_config(args.config)
    paths = get_data_paths(config)
    proc_params = get_processing_params(config)
    quality_config = config.get('quality_filters', {})
    fap_config = config.get('fap', {})
    min_obs = quality_config.get('min_observations', 40)

    # --- Components ---
    virac = ViracInterface(lc_dir=str(paths['virac_lightcurves']))

    quality_filter = QualityFilter(
        max_chi=quality_config.get('max_chi', 10.0),
        max_ast_res_chisq=quality_config.get('max_ast_res_chisq', 20.0),
        max_magerr_sigma=quality_config.get('max_magerr_sigma', 4.0),
        require_positive_mag=quality_config.get('require_positive_mag', True),
        require_positive_magerr=quality_config.get('require_positive_magerr', True),
    )

    feature_calc = FeatureCalculator()

    fap_calc = None
    fap_model_path = fap_config.get('model_path', '')
    if fap_model_path and Path(fap_model_path).exists() and FAP_AVAILABLE:
        try:
            fap_calc = NeuralNetworkFAP(
                model_path=fap_model_path,
                n_points=fap_config.get('n_points', 200),
                knn_neighbors=fap_config.get('knn_neighbors', 10),
            )
            print(f"FAP calculator loaded from {fap_model_path}")
        except Exception as e:
            print(f"WARNING: FAP calculator failed to load: {e}", file=sys.stderr)

    # --- Read FITS chunk ---
    print(f"Reading FITS table from {args.fits}...")
    tbl = Table.read(args.fits, hdu=1)
    end_idx = min(args.start + args.count, len(tbl))
    chunk = tbl[args.start:end_idx]
    source_ids = chunk['sourceid'].data.tolist()
    print(f"Processing {len(source_ids)} sources (indices {args.start} to {end_idx - 1})...")

    # --- Output paths ---
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = output_dir / f"{args.output}.csv"
    fits_path = output_dir / f"{args.output}.fits"

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    # --- Process sources one at a time, append CSV immediately ---
    n_success = 0
    n_total = len(source_ids)

    for i, sid in enumerate(source_ids):
        result = process_single_source(sid, virac, quality_filter, feature_calc, fap_calc, min_obs)

        if result is not None:
            csv_row = [result.get(c, '') for c in COL_NAMES]

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(COL_NAMES)
                    write_header = False
                writer.writerow(csv_row)

            n_success += 1

        if (i + 1) % 100 == 0 or (i + 1) == n_total:
            print(f"  Progress: {i + 1}/{n_total} sources, {n_success} successful")

    print(f"\nCSV done: {n_success}/{n_total} sources written to {csv_path}")

    # --- Write FITS from the completed CSV ---
    if n_success > 0 and csv_path.exists():
        csv_to_fits(str(csv_path), str(fits_path))

    print("Complete.")


if __name__ == "__main__":
    main()
