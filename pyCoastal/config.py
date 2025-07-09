"""
Configuration loader for pyCoastal.
Supports YAML, JSON, and INI-style files.
"""

import os
import json
import configparser

try:
    import yaml
    YAML_OK = True
except ImportError:
    YAML_OK = False


def load_config(path: str) -> dict:
    """
    Load a configuration file and return a dictionary.
    
    Supported formats:
      - .yaml, .yml  (requires PyYAML)
      - .json
      - .ini, .cfg
    
    Parameters
    ----------
    path : str
        Path to the config file.
    
    Returns
    -------
    dict
        Parsed configuration.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".yaml", ".yml"}:
        if not YAML_OK:
            raise ImportError("PyYAML is required to read YAML files.")
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif ext == ".json":
        with open(path, "r") as f:
            return json.load(f)
    elif ext in {".ini", ".cfg"}:
        parser = configparser.ConfigParser()
        parser.read(path)
        cfg = {}
        for section in parser.sections():
            cfg[section] = dict(parser[section])
        return cfg
    else:
        raise ValueError(f"Unsupported config file extension: {ext}")
