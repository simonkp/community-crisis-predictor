import os
import re
from pathlib import Path

import yaml


def _interpolate_env_vars(value):
    if isinstance(value, str):
        pattern = re.compile(r"\$\{(\w+)\}")
        def replacer(match):
            env_var = match.group(1)
            return os.environ.get(env_var, match.group(0))
        return pattern.sub(replacer, value)
    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    return value


def load_config(config_path: str = "config/default.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    config = _interpolate_env_vars(config)
    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    required_sections = ["reddit", "collection", "processing", "features",
                         "labeling", "modeling", "evaluation", "paths"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    if not config["reddit"].get("subreddits"):
        raise ValueError("At least one subreddit must be configured")
