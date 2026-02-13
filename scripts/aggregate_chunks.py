import argparse
import pandas as pd
from pathlib import Path
from astropy.table import Table, vstack
import glob

def main():
    parser = argparse.ArgumentParser(description="Aggregate processed catalog chunks")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing chunk files")
    parser.add_argument("--output", type=str, required=True, help="Final output filename")
    parser.add_argument("--format", type=str, default="fits", choices=["fits", "csv"], help="Output format")
    
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    
    # Find all chunk files
    pattern = str(input_dir / f"primvs_chunk_*.{args.format}")
    chunk_files = sorted(glob.glob(pattern))
    
    if not chunk_files:
        print(f"No chunk files found in {input_dir} with format {args.format}")
        return
        
    print(f"Found {len(chunk_files)} chunks. Aggregating...")
    
    if args.format == "fits":
        tables = []
        for f in chunk_files:
            try:
                tables.append(Table.read(f))
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        final_table = vstack(tables)
        final_table.write(f"{args.output}.fits", overwrite=True)
        print(f"Saved aggregated catalog to {args.output}.fits")
        
    elif args.format == "csv":
        dfs = []
        for f in chunk_files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(f"{args.output}.csv", index=False)
        print(f"Saved aggregated catalog to {args.output}.csv")

if __name__ == "__main__":
    main()
