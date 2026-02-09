# Migration Guide: From Old Pipeline to PRIMVS-Pipeline 2.0

This guide helps you migrate from the old PRIMVS scripts to the new clean pipeline.

## Key Differences

### Old Pipeline
- Hardcoded paths throughout code
- Scattered scripts with duplicate code
- Manual process management
- No configuration system
- Location-specific (UHHPC only)
- Mixed threading/multiprocessing

### New Pipeline
- Zero hardcoded paths - all configurable
- Clean modular package structure
- Automated pipeline orchestration
- YAML-based configuration
- Runs anywhere
- Standardized multiprocessing

## Migration Steps

### 1. Update Your Workflow

**Old way:**
```python
# Old scattered approach
from PRIMVS_file import *
from virac import *
import StochiStats

# Hardcoded paths
LC_dir = '/beegfs/car/njm/LC/'
output_dir = '/beegfs/car/njm/output/'

# Manual processing
for source_id in source_ids:
    lc = Virac.fits_open(f'{LC_dir}{source_id}.FITS')
    # ... manual processing ...
```

**New way:**
```python
# New clean approach
from primvs_pipeline import Pipeline, load_config

# Load configuration
config = load_config('config/pipeline_config.yaml')

# Initialize and run
pipeline = Pipeline(config)
catalogue = pipeline.run(source_ids)
```

### 2. Configure Paths

Create `config/pipeline_config.yaml`:

```yaml
data:
  virac_lc_dir: ${VIRAC_ROOT}/data/output/ts_tables/
  virac_meta_dir: ${VIRAC_ROOT}/data/output/agg_tables/
  output_dir: ./output
  models_dir: ./models
```

Set environment variables:
```bash
export VIRAC_ROOT=/path/to/your/virac/data
```

### 3. Update Function Calls

#### Loading Lightcurves

**Old:**
```python
from virac import Virac
lc = Virac.fits_open('/beegfs/car/njm/LC/123456.FITS')
mag = lc["Ks_mag"]
magerr = lc["Ks_emag"]
time = lc["Ks_mjdobs"]
```

**New:**
```python
from primvs_pipeline.data_access import ViracInterface

virac = ViracInterface(lc_dir=config['data']['virac_lc_dir'])
lc = virac.get_lightcurve(123456, filter_band='Ks')
mag = lc['mag']
magerr = lc['magerr']
time = lc['time']
```

#### Quality Filtering

**Old:**
```python
from PRIMVS_file import error_clip_xy
mag, magerr, time = error_clip_xy(mag, magerr, time, chi, ast_res_chisq, sigma=4, err_max=0.5)
```

**New:**
```python
from primvs_pipeline.preprocessing import QualityFilter

filter = QualityFilter(max_chi=10.0, max_ast_res_chisq=20.0)
filtered_lc = filter.apply(lightcurve)
mag = filtered_lc['mag']
magerr = filtered_lc['magerr']
time = filtered_lc['time']
```

#### Feature Calculation

**Old:**
```python
from StochiStats import cody_M, Eta, Stetson_K
# ... many individual imports ...

cody = cody_M(mag, time)
eta = Eta(mag, time)
stetson = Stetson_K(mag, magerr)
# ... many individual calls ...
```

**New:**
```python
from primvs_pipeline.features import calculate_all_features

features = calculate_all_features(mag, magerr, time)
# All features in one dict:
# features['Cody_M'], features['eta'], features['stet_k'], etc.
```

#### FAP Calculation

**Old:**
```python
from NN_FAP import get_model, inference

knn, model = get_model('/beegfs/car/njm/models/final_12l_dp_all')
fap = inference(period, mag, time, knn, model)
```

**New:**
```python
from primvs_pipeline.fap import NeuralNetworkFAP

fap_calc = NeuralNetworkFAP(model_path='models/fap_nn/final_12l_dp_all')
fap = fap_calc.calculate(period, mag, time)
```

### 4. Update Parallel Processing

**Old:**
```python
from multiprocessing import Pool
from threading import Thread
# Mixed approaches, inconsistent

pool = Pool(16)
results = pool.map(process_func, source_ids)
```

**New:**
```python
from primvs_pipeline.utils import parallel_process

results = parallel_process(
    func=process_func,
    items=source_ids,
    n_processes=16,
    show_progress=True
)
```

### 5. Update File Paths

**Old:**
```python
# Hardcoded everywhere
output_file = '/beegfs/car/njm/output/primvs_catalogue.fits'
model_path = '/beegfs/car/njm/models/fap_nn/'
```

**New:**
```python
# From configuration
output_file = config['data']['output_dir'] / 'primvs_catalogue.fits'
model_path = config['data']['models_dir'] / 'fap_nn'
```

### 6. Column Name Changes

Some column names have been standardized:

| Old Name | New Name |
|----------|----------|
| `Cody_M` | `Cody_M` (unchanged) |
| `stetson_K` | `stet_k` |
| `ls_period` | `ls_p` |
| `pdm_period` | `pdm_p` |
| `ce_period` | `ce_p` |

## Complete Example Migration

### Old Script

```python
#!/usr/bin/env python
import sys
sys.path.append('/beegfs/car/njm/scripts/')

from PRIMVS_file import *
from virac import Virac
import StochiStats
from NN_FAP import get_model, inference

# Hardcoded paths
LC_dir = '/beegfs/car/njm/LC/'
output_dir = '/beegfs/car/njm/output/'
model_path = '/beegfs/car/njm/models/final_12l_dp_all'

# Load model
knn, model = get_model(model_path)

# Process sources
results = []
for source_id in [123456, 234567, 345678]:
    lc = Virac.fits_open(f'{LC_dir}{source_id}.FITS')
    mag, magerr, time = error_clip_xy(lc["Ks_mag"], lc["Ks_emag"], lc["Ks_mjdobs"])
    
    # Calculate features
    cody = StochiStats.cody_M(mag, time)
    eta = StochiStats.Eta(mag, time)
    
    # Calculate FAP
    period = 2.5  # example
    fap = inference(period, mag, time, knn, model)
    
    results.append({'sourceid': source_id, 'cody_M': cody, 'eta': eta, 'fap': fap})

# Save
import pandas as pd
df = pd.DataFrame(results)
df.to_csv(f'{output_dir}/catalogue.csv')
```

### New Script

```python
#!/usr/bin/env python
from primvs_pipeline import Pipeline, load_config

# Load configuration
config = load_config('config/pipeline_config.yaml')

# Initialize pipeline
pipeline = Pipeline(config)

# Process sources
source_ids = [123456, 234567, 345678]
periods = [2.5, 2.5, 2.5]  # example periods

# Run pipeline (automatically handles everything)
catalogue = pipeline.run(
    source_ids=source_ids,
    periods=periods,
    output_name='catalogue'
)

# Done! Catalogue saved automatically in configured formats
```

## Benefits of Migration

1. **Portability**: Runs on any system, not just UHHPC
2. **Maintainability**: Clean modular code, easy to understand
3. **Performance**: Optimized parallel processing
4. **Reliability**: Proper error handling and logging
5. **Flexibility**: Easy to customize via configuration
6. **Documentation**: Comprehensive docs and examples
7. **Testing**: Unit tests ensure correctness

## Troubleshooting

### "Module not found" errors

Make sure you've installed the package:
```bash
pip install -e .
```

### "Configuration file not found"

Specify the config file explicitly:
```python
config = load_config('path/to/your/config.yaml')
```

### "VIRAC files not found"

Check your environment variables and config paths:
```bash
echo $VIRAC_ROOT
```

### Different results from old pipeline

The new pipeline uses the same algorithms but with optimizations. Small numerical differences (<1%) are expected due to:
- Improved numerical stability
- Vectorized operations
- Different random seeds (for padding/deletion)

For exact reproducibility, set random seeds:
```python
import numpy as np
np.random.seed(42)
```

## Getting Help

- Check the [Usage Guide](usage.md)
- See [API Documentation](api.md)
- Review [example scripts](../scripts/)
- Open an issue on GitHub

## Next Steps

After migration:
1. Test on a small subset of sources
2. Compare results with old pipeline
3. Run full reprocessing
4. Update any downstream analysis scripts

