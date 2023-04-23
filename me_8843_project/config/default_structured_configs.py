from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import MISSING

cs = ConfigStore.instance()


@dataclass
class WBConfig:
    """Weights and Biases config"""

    # The name of the project on W&B.
    project_name: str = "me_8843_project"
    # Logging entity (like your username or team name)
    entity: str = "jwspaeth"
    # The group ID to assign to the run. Optional to specify.
    group: str = ""
    # The run name to assign to the run. If not specified,
    # W&B will randomly assign a name.
    run_name: str = ""
    video_dir: str = "video_dir"
    video_fps: int = 50  # default for lunar lander
    video_option: bool = False
    video_size: Optional[List[int]] = None
    video_gray_conversion: bool = False


@dataclass
class LoggingConfig:
    episode_log_interval: int = 5
    training_log_interval: int = 5
    log_file: str = "train.log"
    wb: WBConfig = WBConfig()


@dataclass
class ReplayBufferConfig:
    max_length: int = 100000
    true_states: bool = False


@dataclass
class ReloadConfig:
    path: str = "checkpoints"
    reload: bool = False


@dataclass
class EvalConfig:
    test_episode_count: int = 10


@dataclass
class EnvConfig:
    id: str = MISSING
    render_mode: Optional[str] = "rgb_array"


@dataclass
class PolicyConfig:
    _target_: str = MISSING


@dataclass
class EncoderConfig:
    _target_: str = MISSING


@dataclass
class DecoderConfig:
    _target_: str = MISSING


@dataclass
class TransitionModelConfig:
    _target_: str = MISSING


@dataclass
class RewardModelConfig:
    _target_: str = MISSING


@dataclass
class TrainerConfig:
    env_config: EnvConfig = EnvConfig()
    replay_buffer_config: ReplayBufferConfig = ReplayBufferConfig()
    policy_config: PolicyConfig = PolicyConfig()
    encoder_config: EncoderConfig = EncoderConfig()
    decoder_config: DecoderConfig = DecoderConfig()
    transition_model_config: TransitionModelConfig = TransitionModelConfig()
    reward_model_config: RewardModelConfig = RewardModelConfig()
    eval_config: EvalConfig = EvalConfig()
    logging_config: LoggingConfig = LoggingConfig()
    checkpoint_interval: int = -1
    checkpoint_folder: str = "checkpoints"
    num_checkpoints: int = 10
    max_env_episodes: int = 1000
    max_env_steps: Optional[int] = 10000
    obs_size: Optional[List[int]] = None
    obs_gray_conversion: bool = False
    batch_size: int = 32
    max_training_epochs: Optional[int] = 1
    reconstruction_lr: float = 1e-3
    reward_lr: float = 1e-3
    transition_lr: float = 1e-3


@dataclass
class BaseConfig:
    seed: int = 0
    evaluate: bool = False
    verbose: bool = True
    force_torch_single_threaded: bool = False
    trainer_config: TrainerConfig = TrainerConfig()
    reload_config: ReloadConfig = ReloadConfig()


@dataclass
class LunarLanderEnvConfig(EnvConfig):
    id: str = "LunarLander-v2"
    continuous: bool = True
    gravity: float = -10.0
    enable_wind: bool = False


@dataclass
class RandomPolicyConfig(PolicyConfig):
    _target_: str = "me_8843_project.policies.RandomPolicy"


@dataclass
class TrajectoryOptimizerPolicyConfig(PolicyConfig):
    _target_: str = "me_8843_project.policies.TrajectoryOptimizerPolicy"
    threshold_flag: bool = False
    convergence_threshold: float = 1e-3
    lr: float = 1e-3
    constraint_multipler: float = 1.0
    n_planning_steps: int = 1000


@dataclass
class LunarEncoderConfig(EncoderConfig):
    _target_: str = "me_8843_project.models.lunar_lander.Encoder"


@dataclass
class LunarDecoderConfig(DecoderConfig):
    _target_: str = "me_8843_project.models.lunar_lander.Decoder"


@dataclass
class LunarRewardModelConfig(RewardModelConfig):
    _target_: str = "me_8843_project.models.lunar_lander.RewardModel"


@dataclass
class LunarTransitionModelConfig(TransitionModelConfig):
    _target_: str = "me_8843_project.models.lunar_lander.TransitionModel"


# Register structured configs in hydra registry
cs.store(name="base_config", node=BaseConfig)
cs.store(
    package="trainer_config.env_config",
    group="envs",
    name="lunar_lander_config",
    node=LunarLanderEnvConfig,
)
cs.store(
    package="trainer_config.policy_config",
    group="policies",
    name="random_policy_config",
    node=RandomPolicyConfig,
)
cs.store(
    package="trainer_config.policy_config",
    group="policies",
    name="trajectory_optimizer_policy_config",
    node=TrajectoryOptimizerPolicyConfig,
)
cs.store(
    package="trainer_config.encoder_config",
    group="models/encoder",
    name="lunar_encoder_config",
    node=LunarEncoderConfig,
)
cs.store(
    package="trainer_config.decoder_config",
    group="models/decoder",
    name="lunar_decoder_config",
    node=LunarDecoderConfig,
)
cs.store(
    package="trainer_config.reward_model_config",
    group="models/reward_model",
    name="lunar_reward_model_config",
    node=LunarRewardModelConfig,
)
cs.store(
    package="trainer_config.transition_model_config",
    group="models/transition_model",
    name="lunar_transition_model_config",
    node=LunarTransitionModelConfig,
)


def register_hydra_plugin(plugin) -> None:
    """Hydra users should call this function before invoking @hydra.main"""
    Plugins.instance().register(plugin)


class ME8843ProjectConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="me_8843_project",
            path="pkg://me_8843_project/config/",
        )
