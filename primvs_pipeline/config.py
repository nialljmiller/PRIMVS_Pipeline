"""
Configuration Management Module

Handles loading and validation of pipeline configuration from YAML files.
Supports environment variable expansion for portable path configuration.

Author: Niall Miller
Date: 2025-10-21
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


def expand_env_vars(config: Any) -> Any:
    """
    Recursively expand environment variables in configuration.
    
    Supports ${VAR_NAME} syntax for environment variable substitution.
    
    Args:
        config: Configuration dict, string, or other type
        
    Returns:
        Configuration with environment variables expanded
        
    Example:
        >>> config = {"path": "${HOME}/data"}
        >>> expand_env_vars(config)
        {"path": "/home/user/data"}
    """
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        return os.path.expandvars(config)
    else:
        return config


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML file.
                    If None, looks for config/pipeline_config.yaml
                    
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        ConfigurationError: If config file not found or invalid
        
    Example:
        >>> config = load_config('config/pipeline_config.yaml')
        >>> print(config['processing']['n_processes'])
        16
    """
    # Default config path
    if config_path is None:
        package_dir = Path(__file__).parent.parent
        config_path = package_dir / 'config' / 'pipeline_config.yaml'
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in config file: {e}")
    
    # Expand environment variables
    config = expand_env_vars(config)
    
    # Validate required sections
    required_sections = ['data', 'processing', 'features', 'fap']
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ConfigurationError(f"Missing required config sections: {missing}")
    
    logger.info("Configuration loaded successfully")
    return config


def get_data_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Extract and validate data paths from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of Path objects for data directories
        
    Raises:
        ConfigurationError: If required paths are missing
    """
    data_config = config.get('data', {})
    
    paths = {
        'virac_meta': Path(data_config.get('virac_meta_dir', '.')),
        'virac_lightcurves': Path(data_config.get('virac_lc_dir', '.')),
        'output': Path(data_config.get('output_dir', './output')),
        'models': Path(data_config.get('models_dir', './models')),
    }
    
    # Create output directory if it doesn't exist
    paths['output'].mkdir(parents=True, exist_ok=True)
    
    return paths


def get_processing_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract processing parameters from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of processing parameters
    """
    return config.get('processing', {})


def get_feature_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract feature calculation parameters from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of feature parameters
    """
    return config.get('features', {})


def get_fap_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract FAP calculation parameters from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of FAP parameters
    """
    return config.get('fap', {})

