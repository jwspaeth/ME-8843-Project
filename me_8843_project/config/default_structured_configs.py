from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import MISSING

cs = ConfigStore.instance()


@dataclass
class TrainerConfig:
    checkpoint_interval: int = -1
    checkpoint_folder: str = "data/checkpoints"
    num_updates: int = 10000
    num_checkpoints: int = 10
    total_num_steps: float = -1.0
    num_environments: int = 16


@dataclass
class WBConfig:
    """Weights and Biases config"""

    # The name of the project on W&B.
    project_name: str = ""
    # Logging entity (like your username or team name)
    entity: str = ""
    # The group ID to assign to the run. Optional to specify.
    group: str = ""
    # The run name to assign to the run. If not specified,
    # W&B will randomly assign a name.
    run_name: str = ""


@dataclass
class VideoConfig:
    video_dir: str = "video_dir"
    video_fps: int = 10


@dataclass
class LoggingConfig:
    log_interval: int = 10
    log_file: str = "train.log"
    wb: WBConfig = WBConfig()
    vid: VideoConfig = VideoConfig()


@dataclass
class ReloadConfig:
    ckpt_path: str = "data/checkpoints/ckpt.0.pth"
    state: bool = False
    config: bool = False


@dataclass
class EvalConfig:
    # The split to evaluate on
    split: str = "val"
    # The number of time to run each episode through evaluation.
    # Only works when evaluating on all episodes.
    evals_per_ep: int = 1
    video_option: List[str] = field(
        # available options are "disk" and "tensorboard"
        default_factory=list
    )
    test_episode_count: int = -1


@dataclass
class SimConfig:
    pass


@dataclass
class TaskConfig:
    pass


@dataclass
class BaseConfig:
    evaluate: bool = False
    verbose: bool = True
    task: TaskConfig = TaskConfig()
    sim: SimConfig = SimConfig()
    trainer: TrainerConfig = TrainerConfig()
    logging: LoggingConfig = LoggingConfig()
    reload: ReloadConfig = ReloadConfig()
    eval: EvalConfig = EvalConfig()
    # For our use case, the CPU side things are mainly memory copies
    # and nothing of substantive compute. PyTorch has been making
    # more and more memory copies parallel, but that just ends up
    # slowing those down dramatically and reducing our perf.
    # This forces it to be single threaded.  The default
    # value is left as false as it's different from how
    # PyTorch normally behaves, but all configs we provide
    # set it to true and yours likely should too
    force_torch_single_threaded: bool = True


# Register structured configs in hydra registry
cs.store(group="base", name="config_base", node=BaseConfig)


def register_hydra_plugin(plugin) -> None:
    """Hydra users should call this function before invoking @hydra.main"""
    Plugins.instance().register(plugin)


class ME8843ProjectConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="me_8843_project",
            path="pkg://me_8843_project/config/",
        )
