import logging

import gymnasium as gym
import hydra
import numpy as np
import torch
from me_8843_project.core.ReplayBuffer import ReplayBuffer
from me_8843_project.core.WeightsAndBiasesWriter import WeightsAndBiasesWriter

logger = logging.getLogger(__name__)


class Trainer:
    """
    This class is responsible for training and evaluating the models.
    Leverages hydra for configuration management, and uses its instantiate
    function as a registry system for the models and policies.
    """

    def __init__(
        self,
        env_config,
        encoder_config,
        decoder_config,
        transition_model_config,
        reward_model_config,
        policy_config,
        logging_config,
        replay_buffer_config,
        eval_config,
        max_env_steps=None,
        max_env_episodes=1000,
        training_steps=1000,
        ckpt_config=None,
        checkpoint_interval=-1,
        checkpoint_folder="data/checkpoints",
        num_checkpoints=10,
    ):
        # Setup environment
        self.env = gym.make(**env_config)

        # Setup models and place on GPU if available
        self.models = {
            "encoder": hydra.utils.instantiate(encoder_config),
            "decoder": hydra.utils.instantiate(decoder_config),
            "transition_model": hydra.utils.instantiate(transition_model_config),
            "reward_model": hydra.utils.instantiate(reward_model_config),
        }
        if torch.cuda.is_available():
            for model in self.models.values():
                model.cuda()

        # Setup policy
        self.policy = hydra.utils.instantiate(
            policy_config, env=self.env, models=self.models
        )

        # Setup other components
        self.logging_config = logging_config
        self.logger = WeightsAndBiasesWriter(logging_config.wb)
        self.replay_buffer = ReplayBuffer(**replay_buffer_config)
        self.max_env_steps = max_env_steps
        self.max_env_episodes = max_env_episodes
        self.training_steps = training_steps
        self.eval_config = eval_config
        self.ckpt_config = ckpt_config
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_folder = checkpoint_folder
        self.num_checkpoints = num_checkpoints

    def generate_samples(self):
        # Continue training until max_episodes or convergence is reached
        episode_count = 0
        total_step_count = 0
        episode_metrics = {
            "expected_reward": None,
            "episode_length": None,
        }
        while episode_count < self.max_env_episodes:
            logger.info(f"Episode: {episode_count}")
            observation_1, info = self.env.reset()
            if self.env.render_mode is not None:
                observation_1 = self.env.render()

            # Save initial frame
            if self.logger.video_option:
                self.logger.log_image_to_video(image=observation_1)

            # Initialize metrics
            episode_reward_sum = 0

            # Continue episode until max_steps or termination is reached
            step_count = 0
            terminated = False
            while self.episode_active(step_count, terminated):

                # Take action and observe reward and next state
                action = self.policy(observation_1)
                observation_2, reward, terminated, truncated, info = self.env.step(
                    action
                )
                if self.env.render_mode is not None:
                    observation_2 = self.env.render()

                # Save frames into video
                if self.logger.video_option:
                    self.logger.log_image_to_video(image=observation_2)

                # Save sample to replay buffer
                self.replay_buffer.add_sample(
                    observation_1=observation_1,
                    action=action,
                    reward=reward,
                    observation_2=observation_2,
                    terminated=terminated,
                )

                # Set variables for next iteration
                observation_1 = observation_2
                step_count += 1
                total_step_count += 1
                episode_reward_sum += reward

            # Record metrics
            episode_expected_reward = episode_reward_sum / step_count
            episode_metrics["expected_reward"] = episode_expected_reward
            episode_metrics["episode_length"] = step_count

            # Save video to disk and clear buffer
            if self.logger.video_option:
                self.logger.save_and_clear_video(video_name=f"episode_{episode_count}")

            # Log metrics
            if episode_count % self.logging_config.log_interval == 0:
                self.logger.log_metrics(episode_metrics, total_step_count)

            episode_count += 1

    def episode_active(self, step_count, terminated):
        # Inactive if max steps is reached or termination is true
        if self.max_env_steps is not None:
            step_condition = step_count < self.max_env_steps
        else:
            step_condition = True

        return step_condition and not terminated

    def train_models(self):
        for i in range(self.training_steps):
            # Sample a batch from the replay buffer
            batch = self.replay_buffer.sample_batch()

            # Train the models
            self.models["encoder"].train(batch)
            self.models["decoder"].train(batch)
            self.models["transition_model"].train(batch)
            self.models["reward_model"].train(batch)

    def train(self):
        converged = False
        while not converged:
            self.generate_samples()
            self.train_models()

    def eval(self):
        self.generate_samples()

    @classmethod
    def reload_from_ckpt(cls, ckpt_path):
        raise NotImplementedError

    @classmethod
    def merge_ckpt_config(cls, config, ckpt_config):
        raise NotImplementedError
