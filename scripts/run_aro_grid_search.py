#!/usr/bin/env python3
import argparse
from argparse import Namespace
import itertools
import yaml
import mlflow
from typing import Dict, List, Any
from discoclip.train_aro import train_model

def generate_configs(grid_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all possible combinations of parameters from grid config."""
    # Separate parameters that should be varied from fixed parameters
    varying_params = {}
    fixed_params = {}
    
    for key, value in grid_config.items():
        if isinstance(value, list):
            varying_params[key] = value
        else:
            fixed_params[key] = value
    
    # Generate all combinations of varying parameters
    keys = varying_params.keys()
    values = varying_params.values()
    combinations = list(itertools.product(*values))
    
    # Create full configs by combining with fixed parameters
    configs = []
    for combo in combinations:
        config = fixed_params.copy()
        for key, value in zip(keys, combo):
            config[key] = value
        configs.append(config)
    
    return configs

def main():
    parser = argparse.ArgumentParser(description="Run ARO grid search experiments")
    parser.add_argument("--grid-config", type=str, required=True,
                      help="Path to grid search configuration file")
    args = parser.parse_args()
    
    # Load grid search configuration
    with open(args.grid_config, 'r') as f:
        grid_config = yaml.safe_load(f)
    
    # Generate all configurations
    configs = generate_configs(grid_config)
    print(f"Generated {len(configs)} configurations")
    
    mlflow.set_tracking_uri(grid_config['mlflow_uri'])
    mlflow.set_experiment(grid_config['mlflow_experiment'])
    
    with mlflow.start_run(run_name="grid_search_parent") as parent_run:
        mlflow.log_dict(grid_config, "grid_search_config.yaml")
        
        for i, config in enumerate(configs):
            train_model(Namespace(**config), parent_run=parent_run)
            print(f"Completed {i+1}/{len(configs)} experiments")

if __name__ == "__main__":
    main() 