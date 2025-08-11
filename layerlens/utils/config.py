"""
Configuration utilities for LayerLens.
"""

import os
import json
import yaml
import logging
from dataclasses import dataclass, field, asdict

@dataclass
class LayerLensConfig:
    """Configuration for LayerLens."""
    # General settings
    log_level: str = 'INFO'
    log_dir: str = './logs'
    
    # Surrogate model settings
    surrogate_type: str = 'tree'
    max_samples: int = 1000
    fidelity_threshold: float = 0.9
    
    # Monitoring settings
    enable_monitoring: bool = False
    monitoring_interval: int = 60  # seconds
    drift_sensitivity: float = 0.05
    alert_channels: list = field(default_factory=lambda: ['log'])
    
    # Visualization settings
    dashboard_port: int = 8050
    colormap: str = 'viridis'
    
    # Framework specific settings
    framework_specific: dict = field(default_factory=dict)
    
    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, file_path):
        """Save config to file."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.json':
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif ext in ['.yaml', '.yml']:
            with open(file_path, 'w') as f:
                yaml.dump(self.to_dict(), f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    @classmethod
    def load(cls, file_path):
        """Load config from file."""
        if not os.path.exists(file_path):
            logging.warning(f"Config file not found: {file_path}")
            return cls()
        
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.json':
                with open(file_path, 'r') as f:
                    config_dict = json.load(f)
            elif ext in ['.yaml', '.yml']:
                with open(file_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
            
            return cls(**config_dict)
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            return cls()

def get_config():
    """
    Get the LayerLens configuration.
    
    Checks for config in the following order:
    1. Environment variable LAYERLENS_CONFIG_PATH
    2. ./layerlens_config.yaml
    3. ./layerlens_config.json
    4. Default config
    
    Returns:
        LayerLensConfig object
    """
    # Check environment variable
    config_path = os.environ.get('LAYERLENS_CONFIG_PATH')
    if config_path and os.path.exists(config_path):
        return LayerLensConfig.load(config_path)
    
    # Check current directory
    if os.path.exists('./layerlens_config.yaml'):
        return LayerLensConfig.load('./layerlens_config.yaml')
    if os.path.exists('./layerlens_config.json'):
        return LayerLensConfig.load('./layerlens_config.json')
    
    # Return default config
    return LayerLensConfig()

def configure_logging(config=None):
    """
    Configure logging based on configuration.
    
    Args:
        config: LayerLensConfig object or None to use default
    """
    if config is None:
        config = get_config()
    
    # Set up logging
    numeric_level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    # Create log directory if it doesn't exist
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_dir, 'layerlens.log')),
            logging.StreamHandler()
        ]
    )
