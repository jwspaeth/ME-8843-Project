import copy
import json
import logging
import os
from typing import Any, Optional

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


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
        if config.project_name != "":
            wb_kwargs["project"] = config.project_name
        if config.run_name != "":
            wb_kwargs["name"] = config.run_name
        if config.entity != "":
            wb_kwargs["entity"] = config.entity
        if config.group != "":
            wb_kwargs["group"] = config.group
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

        self.video_option = config.video_option
        self.video_dir = config.video_dir
        self.video_fps = config.video_fps
        self.video_size = config.video_size
        self.video_gray_conversion = config.video_gray_conversion
        self.video_frames = []

    def add_scalars(self, log_group, data_dict, step_id):
        log_data_dict = {
            f"{log_group}/{k.replace(' ', '')}": v for k, v in data_dict.items()
        }
        wandb.log(log_data_dict, step=int(step_id))  # type: ignore[attr-defined]

    def add_scalar(self, key, value, step_id):
        wandb.log({key: value}, step=int(step_id))  # type: ignore[attr-defined]

    def log_metrics(self, metrics, step_id):
        self.print_metrics(metrics)
        wandb.log(metrics, step=int(step_id))  # type: ignore[attr-defined]

    def print_metrics(self, metrics):
        pretty_str = ""
        for key, item in metrics.items():
            if isinstance(item, torch.Tensor):
                item = item.squeeze()
            pretty_str += f"- {key}: {item} -"
        logger.info(f"Metrics: {pretty_str}")

    def __enter__(self):
        return self

    def get_run_id(self) -> Optional[str]:
        return self.run.id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.run:
            self.run.finish()

    def log_image_to_video(self, image):
        image = copy.deepcopy(image)
        self.video_frames.append(image)

    def save_video(self, video_name):
        if self.video_frames:
            processed_video = copy.deepcopy(self.video_frames)
            for i in range(len(processed_video)):

                if self.video_size is not None:
                    processed_video[i] = cv2.resize(processed_video[i], self.video_size)

                if self.video_gray_conversion:
                    processed_video[i] = cv2.cvtColor(
                        processed_video[i], cv2.COLOR_RGB2GRAY
                    )
                    processed_video[i] = cv2.cvtColor(
                        processed_video[i], cv2.COLOR_GRAY2RGB
                    )

            os.makedirs(self.video_dir, exist_ok=True)
            videodims = [processed_video[0].shape[1], processed_video[0].shape[0]]
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            video = cv2.VideoWriter(
                f"{self.video_dir}/{video_name}.mp4", fourcc, self.video_fps, videodims
            )
            for i in range(len(self.video_frames)):
                video.write(processed_video[i])
            video.release()
        else:
            raise Exception("No video frames to save.")

    def save_and_clear_video(self, video_name):
        self.save_video(video_name)
        self.clear_video()

    def clear_video(self):
        self.video_frames.clear()
