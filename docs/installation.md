# Installation Guide

This guide covers the installation of the PRIMVS-Pipeline package.

## Requirements

- Python 3.8 or higher
- pip package manager
- (Optional) conda for environment management

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/nialljmiller/PRIMVS-Pipeline.git
cd PRIMVS-Pipeline
```

### 2. Create Virtual Environment (Recommended)

Using conda:
```bash
conda create -n primvs python=3.10
conda activate primvs
```

Or using venv:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Package

Development mode (recommended for development):
```bash
pip install -e .
```

Standard installation:
```bash
pip install .
```

With development dependencies:
```bash
pip install -e ".[dev]"
```

### 4. Verify Installation

```bash
python -c "import primvs_pipeline; print(primvs_pipeline.__version__)"
```

Or run the test command:
```bash
primvs-pipeline test
```

## Configuration

### 1. Set Environment Variables

The pipeline uses environment variables for data paths. Add these to your `.bashrc` or `.zshrc`:

```bash
# VIRAC data location
export VIRAC_ROOT=/path/to/virac/data

# PRIMVS output directory
export PRIMVS_OUTPUT=/path/to/output

# PRIMVS models directory
export PRIMVS_MODELS=/path/to/models
```

### 2. Edit Configuration File

Copy and edit the configuration file:

```bash
cp config/pipeline_config.yaml config/my_config.yaml
# Edit my_config.yaml with your settings
```

### 3. Download Pre-trained Models

Place the FAP neural network model in the models directory:

```
models/
└── fap_nn/
    ├── final_12l_dp_all_model.json
    ├── final_12l_dp_all_model.h5
    └── final_12l_dp_all_model_history.npy
```

## Troubleshooting

### TensorFlow Installation Issues

If TensorFlow installation fails, try:

```bash
pip install tensorflow==2.10.0 --no-cache-dir
```

For Apple Silicon Macs:
```bash
pip install tensorflow-macos tensorflow-metal
```

### Import Errors

If you get import errors, ensure the package is installed in development mode:

```bash
pip install -e .
```

### Permission Errors

On Linux/Mac, you may need to use `sudo` or install in user space:

```bash
pip install --user -e .
```

## Next Steps

- Read the [Configuration Reference](configuration.md)
- See [Usage Examples](usage.md)
- Check the [API Documentation](api.md)

