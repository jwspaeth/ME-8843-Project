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


def execute_exp(config: "DictConfig", run_type: str) -> None:
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

    trainer = Trainer(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


@hydra.main(
    config_path="config",
    config_name="base/default",
)
def main(cfg: "DictConfig"):
    with read_write(cfg):
        OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    execute_exp(cfg, "eval" if cfg.evaluate else "train")


if __name__ == "__main__":
    register_hydra_plugin(ME8843ProjectConfigPlugin)
    main()
