# PRIMVS-Pipeline

**PeRiodic Infrared Multiclass Variable Stars - Catalogue Construction Pipeline**

A clean, portable, and elegantly coded pipeline for constructing the PRIMVS variable star catalogue from VIRAC infrared time-series data.

## Features

- ✅ **Portable**: No hardcoded paths - runs anywhere
- ✅ **Modular**: Clean separation of concerns
- ✅ **Configurable**: YAML-based configuration system
- ✅ **Documented**: Comprehensive docstrings and examples
- ✅ **Tested**: Unit tests for critical components
- ✅ **Performant**: Optimized parallel processing

## Installation

```bash
# Clone the repository
git clone https://github.com/nialljmiller/PRIMVS-Pipeline.git
cd PRIMVS-Pipeline

# Install in development mode
pip install -e .
```

## Quick Start

```python
from primvs_pipeline import Pipeline
from primvs_pipeline.config import load_config

# Load configuration
config = load_config('config/pipeline_config.yaml')

# Initialize pipeline
pipeline = Pipeline(config)

# Run on a list of source IDs
results = pipeline.process_sources(source_ids)

# Save catalogue
pipeline.save_catalogue('primvs_catalogue.fits')
```

## Pipeline Stages

1. **Data Access** - Retrieve lightcurves from VIRAC database
2. **Preprocessing** - Apply quality filters and clean data
3. **Feature Extraction** - Calculate statistical features and find periods
4. **FAP Calculation** - Compute False Alarm Probability using neural network
5. **Aggregation** - Merge results and create final catalogue

## Configuration

All paths and parameters are configured via `config/pipeline_config.yaml`:

```yaml
data:
  virac_root: /path/to/virac/data
  output_dir: ./output
  
processing:
  n_processes: 16
  fap_threshold: 0.2
  min_observations: 40
```

## Documentation

- [Installation Guide](docs/installation.md)
- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api.md)
- [Development Guide](docs/development.md)

## Project Structure

```
PRIMVS-Pipeline/
├── primvs_pipeline/        # Main package
│   ├── data_access/        # VIRAC interface
│   ├── preprocessing/      # Data cleaning
│   ├── features/           # Feature calculation
│   ├── fap/                # FAP calculation
│   ├── aggregation/        # Catalogue building
│   └── utils/              # Utilities
├── config/                 # Configuration files
├── scripts/                # Command-line scripts
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── models/                 # Pre-trained models
```

## License

MIT License - see LICENSE file for details

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{primvs_pipeline,
  author = {Miller, Niall},
  title = {PRIMVS-Pipeline: Variable Star Catalogue Construction},
  year = {2025},
  url = {https://github.com/nialljmiller/PRIMVS-Pipeline}
}
```

## Contact

- **Author**: Niall Miller
- **Email**: [your-email]
- **GitHub**: [@nialljmiller](https://github.com/nialljmiller)

