## Usage Guide

This guide demonstrates how to use the PRIMVS pipeline in various scenarios.

## Basic Usage

### Python API

```python
from primvs_pipeline import Pipeline, load_config

# Load configuration
config = load_config('config/pipeline_config.yaml')

# Initialize pipeline
pipeline = Pipeline(config)

# Process sources
source_ids = [123456, 234567, 345678]
catalogue = pipeline.run(source_ids, output_name='my_catalogue')

# Access results
print(catalogue[['sourceid', 'true_period', 'best_fap']])
```

### Command-Line Interface

Process sources from a file:
```bash
primvs-pipeline run --sources source_ids.txt --output primvs_catalogue
```

Process specific sources:
```bash
primvs-pipeline run --source-ids 123456 234567 345678 --output test_catalogue
```

With custom configuration:
```bash
primvs-pipeline run --config my_config.yaml --sources sources.txt
```

## Advanced Usage

### Processing with Known Periods

If you have known periods for your sources:

```python
source_ids = [123456, 234567, 345678]
periods = [2.5, 3.7, 1.2]  # days

catalogue = pipeline.process_sources(
    source_ids=source_ids,
    periods=periods,
    n_processes=16
)
```

Or from files:
```bash
primvs-pipeline run \
    --sources sources.txt \
    --periods periods.txt \
    --output catalogue
```

### Custom Quality Filters

Modify quality filters in your configuration:

```yaml
quality_filters:
  min_observations: 50
  max_chi: 8.0
  max_ast_res_chisq: 15.0
  max_magerr_sigma: 3.0
```

Or programmatically:

```python
from primvs_pipeline.preprocessing import QualityFilter

# Create custom filter
quality_filter = QualityFilter(
    max_chi=8.0,
    max_ast_res_chisq=15.0,
    max_magerr_sigma=3.0
)

# Apply to lightcurve
filtered_lc = quality_filter.apply(lightcurve)
```

### Processing Individual Components

You can use individual pipeline components separately:

#### Load Lightcurve

```python
from primvs_pipeline.data_access import ViracInterface

virac = ViracInterface(lc_dir='/path/to/virac/lightcurves')
lc = virac.get_lightcurve(source_id=123456, filter_band='Ks')

mag = lc['mag']
magerr = lc['magerr']
time = lc['time']
```

#### Calculate Features

```python
from primvs_pipeline.features import calculate_all_features

features = calculate_all_features(mag, magerr, time)
print(f"Cody M: {features['Cody_M']}")
print(f"Eta: {features['eta']}")
```

#### Calculate FAP

```python
from primvs_pipeline.fap import NeuralNetworkFAP

fap_calc = NeuralNetworkFAP(model_path='models/fap_nn')
fap = fap_calc.calculate(period=2.5, mag=mag, time=time)
print(f"FAP: {fap:.4f}")
```

### Parallel Processing

Control parallelization:

```python
# Use all available cores
catalogue = pipeline.process_sources(source_ids, n_processes=-1)

# Use specific number of cores
catalogue = pipeline.process_sources(source_ids, n_processes=16)

# Single-threaded (for debugging)
catalogue = pipeline.process_sources(source_ids, n_processes=1)
```

### Saving Catalogues

Save in multiple formats:

```python
# FITS format (default)
pipeline.save_catalogue(catalogue, 'primvs_cat', formats=['fits'])

# CSV format
pipeline.save_catalogue(catalogue, 'primvs_cat', formats=['csv'])

# Both formats
pipeline.save_catalogue(catalogue, 'primvs_cat', formats=['fits', 'csv'])

# HDF5 format
pipeline.save_catalogue(catalogue, 'primvs_cat', formats=['hdf5'])
```

### Loading Catalogues

```python
from primvs_pipeline.data_access import load_catalogue

# Auto-detect format from extension
catalogue = load_catalogue('primvs_catalogue.fits')

# Specify format explicitly
catalogue = load_catalogue('primvs_catalogue.csv', format='csv')
```

## Batch Processing

For processing large numbers of sources:

```python
import numpy as np

# Load all source IDs
with open('all_sources.txt', 'r') as f:
    all_source_ids = [int(line.strip()) for line in f]

# Process in chunks
chunk_size = 10000
n_chunks = len(all_source_ids) // chunk_size + 1

for i in range(n_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(all_source_ids))
    
    chunk_ids = all_source_ids[start:end]
    
    print(f"Processing chunk {i+1}/{n_chunks} ({len(chunk_ids)} sources)")
    
    catalogue_chunk = pipeline.process_sources(chunk_ids)
    
    # Save chunk
    pipeline.save_catalogue(
        catalogue_chunk,
        output_name=f'primvs_catalogue_chunk_{i:04d}'
    )
```

## Logging

Control logging verbosity:

```python
from primvs_pipeline.utils.logging_config import setup_logging

# Debug level (very verbose)
logger = setup_logging(level='DEBUG', log_file='pipeline_debug.log')

# Info level (default)
logger = setup_logging(level='INFO')

# Warning level (quiet)
logger = setup_logging(level='WARNING')
```

Or via CLI:
```bash
primvs-pipeline run --sources sources.txt --log-level DEBUG --log-file pipeline.log
```

## Error Handling

The pipeline handles errors gracefully:

```python
# Sources that fail processing return None
# They are automatically filtered from results
catalogue = pipeline.process_sources(source_ids)

# Check how many sources succeeded
n_success = len(catalogue)
n_total = len(source_ids)
print(f"Success rate: {n_success}/{n_total} ({n_success/n_total*100:.1f}%)")
```

## Performance Tips

1. **Use parallelization**: Set `n_processes` to match your CPU cores
2. **Process in chunks**: For very large datasets, process in chunks to manage memory
3. **Enable caching**: Set `use_cache: true` in configuration
4. **Use SSD storage**: Place VIRAC data on fast storage
5. **Increase chunk_size**: For many small files, increase `chunk_size` in config

## Next Steps

- See [Configuration Reference](configuration.md) for all options
- Check [API Documentation](api.md) for detailed API reference
- Read [Development Guide](development.md) for contributing

