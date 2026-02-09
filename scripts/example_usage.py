#!/usr/bin/env python3
"""
Example Usage Script

Demonstrates how to use the PRIMVS pipeline.

Author: Niall Miller
Date: 2025-10-21
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from primvs_pipeline import Pipeline, load_config
from primvs_pipeline.utils.logging_config import setup_logging


def main():
    """Run example pipeline."""
    # Setup logging
    logger = setup_logging(level="INFO")
    
    logger.info("PRIMVS Pipeline Example")
    logger.info("=" * 60)
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = Pipeline(config)
    
    # Example source IDs (replace with real ones)
    source_ids = [
        123456,
        234567,
        345678,
        456789,
        567890,
    ]
    
    # Optional: provide known periods
    periods = [
        2.5,
        3.7,
        1.2,
        5.4,
        0.8,
    ]
    
    logger.info(f"Processing {len(source_ids)} sources...")
    
    # Process sources
    catalogue = pipeline.process_sources(
        source_ids=source_ids,
        periods=periods,
        n_processes=4,
        show_progress=True
    )
    
    # Display results
    logger.info(f"\nProcessed {len(catalogue)} sources")
    logger.info(f"Features calculated: {len(catalogue.columns)}")
    
    if len(catalogue) > 0:
        logger.info("\nSample of results:")
        logger.info(catalogue[['sourceid', 'mag_n', 'true_amplitude', 'best_fap']].head())
        
        # Save catalogue
        logger.info("\nSaving catalogue...")
        pipeline.save_catalogue(catalogue, output_name="example_catalogue")
        logger.info("Done!")
    else:
        logger.warning("No sources were successfully processed")


if __name__ == '__main__':
    main()

