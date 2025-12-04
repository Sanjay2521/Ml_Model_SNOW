"""Configuration utilities"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Configuration loaded from: {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        config_path: Output file path
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to: {config_path}")


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from JSON file

    Args:
        file_path: Path to JSON file

    Returns:
        Data dictionary
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    return data


def save_json(data: Dict[str, Any], file_path: str):
    """
    Save data to JSON file

    Args:
        data: Data dictionary
        file_path: Output file path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Data saved to: {file_path}")


def update_config(config_path: str, updates: Dict[str, Any]):
    """
    Update configuration file

    Args:
        config_path: Path to configuration file
        updates: Dictionary of updates
    """
    config = load_config(config_path)

    # Deep update
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    config = deep_update(config, updates)
    save_config(config, config_path)

    print(f"Configuration updated")


if __name__ == "__main__":
    # Example usage
    config = {
        'data': {
            'path': 'data/raw',
            'target': 'label'
        },
        'model': {
            'type': 'random_forest',
            'params': {
                'n_estimators': 100
            }
        }
    }

    save_config(config, 'test_config.yaml')
    loaded_config = load_config('test_config.yaml')
    print(loaded_config)
