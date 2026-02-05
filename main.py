import logging
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    from stochopt.runner import run_experiment

    log = logging.getLogger(__name__)
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    run_experiment(cfg)


if __name__ == "__main__":
    main()
