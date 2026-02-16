import argparse
import os

# Force CPU-only TF before anything imports TensorFlow
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from pathlib import Path
from astropy.table import Table
from primvs_pipeline import Pipeline, load_config

def main():
    parser = argparse.ArgumentParser(description="Process a chunk of the PRIMVS catalog")
    parser.add_argument("--fits", type=str, required=True, help="Path to the catalog FITS file")
    parser.add_argument("--start", type=int, default=0, help="Starting index in the FITS table")
    parser.add_argument("--count", type=int, default=1000, help="Number of sources to process")
    parser.add_argument("--output", type=str, required=True, help="Output filename base")
    parser.add_argument("--n-processes", type=int, default=-1, help="Number of parallel processes")
    parser.add_argument("--config", type=str, default="../config/pipeline_config.yaml", help="Path to config file")
    parser.add_argument("--recompute-periods", action="store_true", default=True,
                        help="Recompute periods from scratch (default: True)")
    parser.add_argument("--use-existing-periods", action="store_true", default=False,
                        help="Use existing periods from the FITS table instead of recomputing")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if args.n_processes != -1:
        config['processing']['n_processes'] = args.n_processes
        
    # Initialize pipeline
    pipeline = Pipeline(config)
    
    # Read the FITS table chunk
    print(f"Reading FITS table from {args.fits}...")
    tbl = Table.read(args.fits, hdu=1)
    
    end_idx = min(args.start + args.count, len(tbl))
    chunk_tbl = tbl[args.start:end_idx]
    
    source_ids = chunk_tbl['sourceid'].data.tolist()
    
    # Handle periods
    periods = None
    if args.use_existing_periods and not args.recompute_periods:
        # Try to extract existing periods from the FITS table
        for period_col in ['true_period', 'period', 'ls_period1']:
            if period_col in chunk_tbl.colnames:
                periods = chunk_tbl[period_col].data.tolist()
                print(f"Using existing periods from column '{period_col}'")
                break
        if periods is None:
            print("No period column found in FITS table, will compute from scratch")
    else:
        print("Computing periods from scratch using LS, PDM, CE, GP periodograms")
    
    print(f"Processing {len(source_ids)} sources (indices {args.start} to {end_idx-1})...")
    
    # Run the pipeline on the chunk
    catalogue = pipeline.run(source_ids, periods, output_name=args.output)
    
    print(f"Successfully processed {len(catalogue)} sources. Results saved to {args.output}")

if __name__ == "__main__":
    main()
