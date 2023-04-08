import os
from typing import Any, Optional

import numpy as np
from omegaconf import OmegaConf

try:
    import wandb
except ImportError:
    wandb = None


class WeightsAndBiasesWriter:
    def __init__(
        self,
        config,
        *args: Any,
        resume_run_id: Optional[str] = None,
        **kwargs: Any,
    ):
        r"""
        Integrates with https://wandb.ai logging service.
        """
        wb_kwargs = {}
        if config.wb.project_name != "":
            wb_kwargs["project"] = config.wb.project_name
        if config.wb.run_name != "":
            wb_kwargs["name"] = config.wb.run_name
        if config.wb.entity != "":
            wb_kwargs["entity"] = config.wb.entity
        if config.wb.group != "":
            wb_kwargs["group"] = config.wb.group
        slurm_info_dict = {
            k[len("SLURM_") :]: v
            for k, v in os.environ.items()
            if k.startswith("SLURM_")
        }
        if wandb is None:
            raise ValueError("Requested to log with wandb, but wandb is not installed.")
        if resume_run_id is not None:
            wb_kwargs["id"] = resume_run_id
            wb_kwargs["resume"] = "must"

        self.run = wandb.init(  # type: ignore[attr-defined]
            config={
                "slurm": slurm_info_dict,
                **OmegaConf.to_container(config),  # type: ignore[arg-type]
            },
            **wb_kwargs,
        )

    def __getattr__(self, item):
        if self.writer:
            return self.writer.__getattribute__(item)
        else:
            return lambda *args, **kwargs: None

    def add_scalars(self, log_group, data_dict, step_id):
        log_data_dict = {
            f"{log_group}/{k.replace(' ', '')}": v for k, v in data_dict.items()
        }
        wandb.log(log_data_dict, step=int(step_id))  # type: ignore[attr-defined]

    def add_scalar(self, key, value, step_id):
        wandb.log({key: value}, step=int(step_id))  # type: ignore[attr-defined]

    def __enter__(self):
        return self

    def get_run_id(self) -> Optional[str]:
        return self.run.id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.run:
            self.run.finish()

    def add_video_from_np_images(
        self, video_name: str, step_idx: int, images: np.ndarray, fps: int = 10
    ) -> None:
        raise NotImplementedError("Not supported")
