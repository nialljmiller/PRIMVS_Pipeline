"""
Process a chunk of the PRIMVS catalog with parallel processing and
row-by-row CSV append.

Each batch of sources is processed in parallel using joblib, then
results are appended to the output CSV. This ensures partial results
are saved even if the job is killed, while utilising all available cores.

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
import time as pytime
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.table import Table
from joblib import Parallel, delayed

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
    sourceid_trimmed = sourceid_str[:-4]
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
        r['_ls_ok'] = True
    except Exception as e:
        r['_ls_ok'] = False
        r['_ls_err'] = str(e)
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
        r['_pdm_ok'] = True
    except Exception as e:
        r['_pdm_ok'] = False
        r['_pdm_err'] = str(e)
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
        r['_ce_ok'] = True
    except Exception as e:
        r['_ce_ok'] = False
        r['_ce_err'] = str(e)
    return r


def run_gp(mag, magerr, time):
    r = {}
    try:
        gp = GP(mag, magerr, time)
        r['gp_lnlike'] = gp.get('log_likelihood', np.nan)
        r['gp_b']      = gp.get('b', np.nan)
        r['gp_c']      = gp.get('c', np.nan)
        r['gp_p']      = gp.get('period', np.nan)
        r['_gp_ok'] = True
    except Exception as e:
        r['_gp_ok'] = False
        r['_gp_err'] = str(e)
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
# Single-source processing (runs in worker processes)
# ---------------------------------------------------------------------------

def process_single_source(source_id, lc_dir, quality_kwargs, fap_config_dict, min_obs=40):
    """
    Process one source through the full pipeline.
    Returns a dict keyed by COL_NAMES, or None on failure.

    NOTE: Each worker re-instantiates lightweight components to avoid
    pickling issues with joblib. The heavy objects (ViracInterface,
    QualityFilter, FeatureCalculator) are cheap to construct.
    """
    t0 = pytime.time()
    row = {c: np.nan for c in COL_NAMES}
    row['sourceid'] = int(source_id)
    row['uniqueid'] = 0
    row['true_class'] = ''
    row['best_method'] = ''
    row['trans_flag'] = 0.0

    # Diagnostic metadata (stripped before CSV write)
    diag = {
        'sourceid': int(source_id),
        'status': 'unknown',
        'n_obs_raw': 0,
        'n_obs_filtered': 0,
        'methods': {'ls': None, 'pdm': None, 'ce': None, 'gp': None},
        'best_period': np.nan,
        'best_fap': np.nan,
        'best_method': '',
        'elapsed_s': 0.0,
        'error': None,
    }

    try:
        # Reconstruct lightweight components in worker
        virac = ViracInterface(lc_dir=lc_dir)
        quality_filter = QualityFilter(**quality_kwargs)
        feature_calc = FeatureCalculator()

        fap_calc = None
        fap_model_path = fap_config_dict.get('model_path', '')
        if fap_model_path and Path(fap_model_path).exists() and FAP_AVAILABLE:
            try:
                fap_calc = NeuralNetworkFAP(
                    model_path=fap_model_path,
                    n_points=fap_config_dict.get('n_points', 200),
                    knn_neighbors=fap_config_dict.get('knn_neighbors', 10),
                )
            except Exception:
                pass

        # 1. Load lightcurve
        lc = virac.get_lightcurve(source_id, filter_band='Ks')
        diag['n_obs_raw'] = len(lc.get('mag', []))

        # 2. Quality filter
        flc = quality_filter.apply(lc)
        mag    = flc['mag']
        magerr = flc['magerr']
        time   = flc['time']
        diag['n_obs_filtered'] = len(mag)

        if len(mag) < min_obs:
            diag['status'] = 'skipped_too_few_obs'
            diag['elapsed_s'] = pytime.time() - t0
            return None, diag

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
            ls_res = run_ls(mag, magerr, time)
            diag['methods']['ls'] = 'ok' if ls_res.pop('_ls_ok', False) else ls_res.pop('_ls_err', 'fail')
            row.update(ls_res)

            pdm_res = run_pdm(mag, magerr, time)
            diag['methods']['pdm'] = 'ok' if pdm_res.pop('_pdm_ok', False) else pdm_res.pop('_pdm_err', 'fail')
            row.update(pdm_res)

            ce_res = run_ce(mag, magerr, time)
            diag['methods']['ce'] = 'ok' if ce_res.pop('_ce_ok', False) else ce_res.pop('_ce_err', 'fail')
            row.update(ce_res)

            gp_res = run_gp(mag, magerr, time)
            diag['methods']['gp'] = 'ok' if gp_res.pop('_gp_ok', False) else gp_res.pop('_gp_err', 'fail')
            row.update(gp_res)

        # 5. FAP + best period
        row = compute_fap_and_select_best(row, fap_calc, mag, time)

        # 6. Generate PRIMVS ID
        true_period = row.get('true_period', 0.0)
        best_fap    = row.get('best_fap', 1.0)
        if np.isnan(true_period):
            true_period = 0.0
        if np.isnan(best_fap):
            best_fap = 1.0
        row['uniqueid'] = generate_primvs_id(source_id, true_period, best_fap)

        diag['status'] = 'success'
        diag['best_period'] = row['true_period']
        diag['best_fap'] = row['best_fap']
        diag['best_method'] = row['best_method']
        diag['elapsed_s'] = pytime.time() - t0
        return row, diag

    except FileNotFoundError:
        diag['status'] = 'no_lightcurve'
        diag['elapsed_s'] = pytime.time() - t0
        return None, diag
    except Exception as e:
        diag['status'] = 'error'
        diag['error'] = str(e)
        diag['elapsed_s'] = pytime.time() - t0
        return None, diag


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
# Logging helpers
# ---------------------------------------------------------------------------

def format_eta(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def print_batch_summary(batch_diags, batch_num, total_batches,
                        cum_success, cum_fail, cum_skip, cum_total, job_t0):
    """Print a detailed summary after each parallel batch completes."""
    elapsed = pytime.time() - job_t0
    rate = cum_total / elapsed if elapsed > 0 else 0
    remaining = (total_batches - batch_num) * (elapsed / batch_num) if batch_num > 0 else 0

    # Per-batch breakdown
    b_success = sum(1 for d in batch_diags if d['status'] == 'success')
    b_skip = sum(1 for d in batch_diags if d['status'] in ('skipped_too_few_obs', 'no_lightcurve'))
    b_fail = sum(1 for d in batch_diags if d['status'] == 'error')
    b_times = [d['elapsed_s'] for d in batch_diags]
    avg_time = np.mean(b_times) if b_times else 0

    # Method success rates for this batch
    method_counts = {'ls': 0, 'pdm': 0, 'ce': 0, 'gp': 0}
    method_ok = {'ls': 0, 'pdm': 0, 'ce': 0, 'gp': 0}
    for d in batch_diags:
        if d['status'] == 'success':
            for m in ['ls', 'pdm', 'ce', 'gp']:
                method_counts[m] += 1
                if d['methods'][m] == 'ok':
                    method_ok[m] += 1

    sep = "-" * 72
    print(sep)
    print(f"  BATCH {batch_num}/{total_batches}  |  "
          f"Elapsed: {format_eta(elapsed)}  |  "
          f"ETA: {format_eta(remaining)}  |  "
          f"Rate: {rate:.1f} src/s")
    print(f"  This batch:  {b_success} ok / {b_skip} skipped / {b_fail} errors  "
          f"(avg {avg_time:.2f}s/source)")
    print(f"  Cumulative:  {cum_success} ok / {cum_skip} skipped / {cum_fail} errors  "
          f"({cum_total} total)")

    if sum(method_counts.values()) > 0:
        method_str = "  Methods:    "
        for m in ['ls', 'pdm', 'ce', 'gp']:
            if method_counts[m] > 0:
                pct = 100 * method_ok[m] / method_counts[m]
                method_str += f" {m.upper()}={pct:.0f}%"
            else:
                method_str += f" {m.upper()}=n/a"
        print(method_str)

    # Report any errors
    errors = [(d['sourceid'], d['error']) for d in batch_diags if d['status'] == 'error']
    if errors:
        print(f"  Errors ({len(errors)}):")
        for sid, err in errors[:5]:
            print(f"    sourceid {sid}: {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")
    print(sep, flush=True)


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
                        help="Output name (writes output/<n>.csv and output/<n>.fits)")
    parser.add_argument("--config", type=str, default="../config/pipeline_config.yaml")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0 = use all available CPUs)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Sources per parallel batch (0 = auto, workers * 2)")
    args = parser.parse_args()

    n_workers = args.workers if args.workers > 0 else os.cpu_count()
    batch_size = args.batch_size if args.batch_size > 0 else n_workers * 2

    print("=" * 72)
    print(f"  PRIMVS Chunk Processor")
    print(f"  Workers: {n_workers}  |  Batch size: {batch_size}")
    print("=" * 72, flush=True)

    # --- Config ---
    config = load_config(args.config)
    paths = get_data_paths(config)
    proc_params = get_processing_params(config)
    quality_config = config.get('quality_filters', {})
    fap_config = config.get('fap', {})
    min_obs = quality_config.get('min_observations', 40)

    # Serialisable kwargs for workers (avoid pickling complex objects)
    lc_dir = str(paths['virac_lightcurves'])
    quality_kwargs = {
        'max_chi': quality_config.get('max_chi', 10.0),
        'max_ast_res_chisq': quality_config.get('max_ast_res_chisq', 20.0),
        'max_magerr_sigma': quality_config.get('max_magerr_sigma', 4.0),
        'require_positive_mag': quality_config.get('require_positive_mag', True),
        'require_positive_magerr': quality_config.get('require_positive_magerr', True),
    }
    fap_config_dict = dict(fap_config)

    print(f"Lightcurve dir: {lc_dir}")
    print(f"Quality filter: min_obs={min_obs}, max_chi={quality_kwargs['max_chi']}")
    print(f"Periodograms available: {PERIODOGRAMS_AVAILABLE}")
    print(f"FAP available: {FAP_AVAILABLE}")
    print(f"Cody_Q available: {CODY_Q_AVAILABLE}")

    # --- Read FITS chunk ---
    print(f"\nReading FITS table from {args.fits}...")
    tbl = Table.read(args.fits, hdu=1)
    end_idx = min(args.start + args.count, len(tbl))
    chunk = tbl[args.start:end_idx]
    source_ids = chunk['sourceid'].data.tolist()
    print(f"Loaded {len(source_ids)} sources (indices {args.start} to {end_idx - 1})")

    # --- Output paths ---
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = output_dir / f"{args.output}.csv"
    fits_path = output_dir / f"{args.output}.fits"

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    # --- Process in parallel batches ---
    n_total = len(source_ids)
    n_batches = (n_total + batch_size - 1) // batch_size
    cum_success = 0
    cum_fail = 0
    cum_skip = 0
    job_t0 = pytime.time()

    print(f"\nProcessing {n_total} sources in {n_batches} batches of ~{batch_size}...")
    print(flush=True)

    for batch_idx in range(n_batches):
        b_start = batch_idx * batch_size
        b_end = min(b_start + batch_size, n_total)
        batch_sids = source_ids[b_start:b_end]

        # Run batch in parallel
        results = Parallel(n_jobs=n_workers, prefer="processes")(
            delayed(process_single_source)(
                sid, lc_dir, quality_kwargs, fap_config_dict, min_obs
            )
            for sid in batch_sids
        )

        # Collect results and append to CSV
        batch_diags = []
        batch_rows = []
        for row, diag in results:
            batch_diags.append(diag)
            if row is not None:
                batch_rows.append(row)
                cum_success += 1
            elif diag['status'] == 'error':
                cum_fail += 1
            else:
                cum_skip += 1

        # Write batch to CSV
        if batch_rows:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(COL_NAMES)
                    write_header = False
                for row in batch_rows:
                    writer.writerow([row.get(c, '') for c in COL_NAMES])

        cum_total = cum_success + cum_fail + cum_skip
        print_batch_summary(
            batch_diags, batch_idx + 1, n_batches,
            cum_success, cum_fail, cum_skip, cum_total, job_t0
        )

    # --- Final summary ---
    total_elapsed = pytime.time() - job_t0
    print(f"\n{'=' * 72}")
    print(f"  COMPLETE")
    print(f"  Total time: {format_eta(total_elapsed)}")
    print(f"  Sources: {cum_success} successful / {cum_skip} skipped / {cum_fail} errors")
    print(f"  Avg rate: {(cum_success + cum_fail + cum_skip) / total_elapsed:.1f} sources/sec")
    print(f"  CSV: {csv_path}")

    # --- Write FITS from the completed CSV ---
    if cum_success > 0 and csv_path.exists():
        csv_to_fits(str(csv_path), str(fits_path))
        print(f"  FITS: {fits_path}")

    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()