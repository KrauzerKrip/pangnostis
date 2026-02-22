import yaml
from dataclasses import dataclass
from typing import Dict, Optional
import os

@dataclass
class Prompt:
    name: str
    system: str
    user: str

def load_prompts(config_path: str = "config/prompts.yaml") -> Dict[str, Prompt]:
    """
    Loads prompt configurations from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Prompt]: A dictionary mapping prompt names to Prompt objects.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    prompts = {}
    if not data:
        return prompts

    for name, config in data.items():
        # Ensure we have the required fields
        if not isinstance(config, dict):
            # skipping invalid entries or handle error? 
            # For now, let's assume valid structure or let it fail if keys missing
            continue
            
        system_template = config.get("system", "")
        user_template = config.get("user", "")
        
        prompts[name] = Prompt(
            name=name,
            system=system_template,
            user=user_template
        )

    return prompts