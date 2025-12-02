import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.runner import run_experiment

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    run_experiment(cfg)

if __name__ == "__main__":
    main()
