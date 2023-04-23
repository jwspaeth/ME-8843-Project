import logging
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, read_write

from me_8843_project.config.default_structured_configs import (
    ME8843ProjectConfigPlugin,
    register_hydra_plugin,
)
from me_8843_project.core.Trainer import Trainer

logger = logging.getLogger(__name__)


def load_trainer(config: "DictConfig") -> Trainer:
    r"""Load trainer using config, possibly reloading from checkpoint
    Args:
        config: DictConfig
    """
    if config.reload_config.reload:
        logger.info("Reloading trainer from checkpoint")
        trainer = Trainer.reload_from_ckpt(config)
    else:
        logger.info("Creating new trainer")
        trainer = Trainer(**config.trainer_config)

    return trainer


def save_config(cfg: "DictConfig") -> None:
    r"""Save config to file
    Args:
    cfg: DictConfig
    path: str
    """
    os.makedirs(cfg.trainer_config.checkpoint_folder, exist_ok=True)

    path = os.path.join(cfg.trainer_config.checkpoint_folder, "config.yaml")
    with open(path, "w") as f:
        OmegaConf.save(cfg, f)


def execute_exp(config: "DictConfig") -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: DictConfig
    runtype: str {train or eval}
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.force_torch_single_threaded and torch.cuda.is_available():
        torch.set_num_threads(1)

    # Create trainer
    trainer = load_trainer(config)

    if not config.evaluate:
        trainer.train()
    else:
        trainer.eval()


@hydra.main(version_base="1.3", config_path="config/base", config_name="lunar_lander")
def main(cfg: "DictConfig"):
    torch.autograd.set_detect_anomaly(True)

    # Resolve and print config
    with read_write(cfg):
        OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    # Save config to file
    save_config(cfg)

    # Execute experiment
    execute_exp(cfg)


if __name__ == "__main__":
    register_hydra_plugin(ME8843ProjectConfigPlugin)
    main()
