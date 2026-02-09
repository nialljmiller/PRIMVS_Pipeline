"""
Command-Line Interface Module

Provides CLI for running the PRIMVS pipeline.

Author: Niall Miller
Date: 2025-10-21
"""

import argparse
import sys
from pathlib import Path
import logging

from .pipeline import Pipeline
from .config import load_config
from .utils.logging_config import setup_logging


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PRIMVS Pipeline - Variable Star Catalogue Construction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process sources from a file
  primvs-pipeline run --sources sources.txt --output primvs_catalogue
  
  # Process with custom config
  primvs-pipeline run --config my_config.yaml --sources sources.txt
  
  # Process specific sources
  primvs-pipeline run --source-ids 123456 789012 345678
        """
    )
    
    parser.add_argument(
        'command',
        choices=['run', 'test'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--sources',
        type=str,
        default=None,
        help='Path to file containing source IDs (one per line)'
    )
    
    parser.add_argument(
        '--source-ids',
        type=int,
        nargs='+',
        default=None,
        help='List of source IDs to process'
    )
    
    parser.add_argument(
        '--periods',
        type=str,
        default=None,
        help='Path to file containing periods (one per line, matching sources)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='primvs_catalogue',
        help='Output catalogue name (without extension)'
    )
    
    parser.add_argument(
        '--n-processes',
        type=int,
        default=None,
        help='Number of parallel processes (-1 for all cores)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Log file path'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(
        level=args.log_level,
        log_file=args.log_file
    )
    
    logger.info("PRIMVS Pipeline CLI")
    logger.info(f"Command: {args.command}")
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Override n_processes if specified
    if args.n_processes is not None:
        config['processing']['n_processes'] = args.n_processes
    
    # Execute command
    if args.command == 'run':
        run_pipeline(args, config, logger)
    elif args.command == 'test':
        test_pipeline(args, config, logger)


def run_pipeline(args, config, logger):
    """Run the pipeline."""
    # Get source IDs
    if args.source_ids is not None:
        source_ids = args.source_ids
        logger.info(f"Processing {len(source_ids)} sources from command line")
    elif args.sources is not None:
        source_file = Path(args.sources)
        if not source_file.exists():
            logger.error(f"Source file not found: {source_file}")
            sys.exit(1)
        
        with open(source_file, 'r') as f:
            source_ids = [int(line.strip()) for line in f if line.strip()]
        
        logger.info(f"Loaded {len(source_ids)} sources from {source_file}")
    else:
        logger.error("Must specify either --sources or --source-ids")
        sys.exit(1)
    
    # Get periods if provided
    periods = None
    if args.periods is not None:
        period_file = Path(args.periods)
        if not period_file.exists():
            logger.error(f"Period file not found: {period_file}")
            sys.exit(1)
        
        with open(period_file, 'r') as f:
            periods = [float(line.strip()) for line in f if line.strip()]
        
        if len(periods) != len(source_ids):
            logger.error(f"Number of periods ({len(periods)}) does not match number of sources ({len(source_ids)})")
            sys.exit(1)
        
        logger.info(f"Loaded {len(periods)} periods from {period_file}")
    
    # Initialize pipeline
    try:
        pipeline = Pipeline(config)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Run pipeline
    try:
        catalogue = pipeline.run(source_ids, periods, output_name=args.output)
        logger.info(f"Pipeline completed successfully - {len(catalogue)} sources in catalogue")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


def test_pipeline(args, config, logger):
    """Test the pipeline with a small sample."""
    logger.info("Running pipeline test...")
    
    # Use a small test set
    test_source_ids = [123456, 234567, 345678]  # Example IDs
    
    logger.info(f"Testing with {len(test_source_ids)} sources")
    
    try:
        pipeline = Pipeline(config)
        catalogue = pipeline.run(test_source_ids, output_name="test_catalogue")
        
        if len(catalogue) > 0:
            logger.info("✓ Pipeline test PASSED")
            logger.info(f"  Processed {len(catalogue)}/{len(test_source_ids)} sources")
            logger.info(f"  Features calculated: {len(catalogue.columns)}")
        else:
            logger.warning("⚠ Pipeline test completed but no sources processed")
    
    except Exception as e:
        logger.error(f"✗ Pipeline test FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

