import logging

import cv2
import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from me_8843_project.core.ObservationQueue import ObservationQueue
from me_8843_project.core.ReplayBuffer import ReplayBuffer
from me_8843_project.core.WeightsAndBiasesWriter import WeightsAndBiasesWriter
from tqdm import tqdm

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
        obs_size=None,
        obs_gray_conversion=False,
        model_learning_rate=1e-3,
        reconstruction_loss_weight=1.0,
        reward_loss_weight=1.0,
        transition_loss_weight=1.0,
        batch_size=32,
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

        # Setup model optimizer
        self.model_learning_rate = model_learning_rate
        params = []
        for values in self.models.values():
            for param in values.parameters():
                params.append(param)
        self.model_optimizer = torch.optim.Adam(params, lr=self.model_learning_rate)

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
        self.obs_size = obs_size
        self.obs_gray_conversion = obs_gray_conversion
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.transition_loss_weight = transition_loss_weight
        self.batch_size = batch_size

        # Setup counters
        self.total_env_step_count = 0
        self.total_training_step_count = 0

    def obs_transform(self, obs):
        if self.obs_size is not None:
            obs = cv2.resize(obs, self.obs_size)

        if self.obs_gray_conversion:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = np.expand_dims(obs, axis=2)

        obs = np.transpose(obs, (2, 0, 1))
        obs = np.expand_dims(obs, axis=0)
        obs = torch.from_numpy(obs).float()

        return obs

    def action_transform(self, action):
        action = np.expand_dims(action, axis=0)
        action = torch.from_numpy(action).float()

        return action

    def reward_transform(self, reward):
        reward = np.expand_dims(reward, axis=0)
        reward = np.expand_dims(reward, axis=0)
        reward = torch.from_numpy(reward).float()

        return reward

    def generate_samples(self):
        logger.info("Generating samples.")

        # Continue training until max_episodes or convergence is reached
        # Queue holds most recent 3 observations, which is required for training
        observation_queue = ObservationQueue(maxlen=3)
        episode_count = 0
        episode_metrics = {
            "expected_reward": None,
            "episode_length": None,
        }
        while episode_count < self.max_env_episodes:
            logger.info(f"Episode: {episode_count}")
            observation, info = self.env.reset()
            if self.env.render_mode is not None:
                observation = self.env.render()

            # Save initial frame
            if self.logger.video_option:
                self.logger.log_image_to_video(image=observation)

            # Transform observations
            observation = self.obs_transform(observation)
            observation_queue.append(observation)

            # Initialize metrics
            episode_reward_sum = 0

            # Continue episode until max_steps or termination is reached
            step_count = 0
            terminated = False
            while self.episode_active(step_count, terminated):

                # Take action and observe reward and next state
                # Must wait for two observations to create state with encoder
                if step_count < 2:
                    action = self.env.action_space.sample()
                else:
                    state = self.models["encoder"](
                        observation_queue[-1], observation_queue[-2]
                    ).detach()
                    action = self.policy(state)
                    # breakpoint()
                observation, reward, terminated, truncated, info = self.env.step(action)
                if self.env.render_mode is not None:
                    observation = self.env.render()

                # Save frames into video
                if self.logger.video_option:
                    self.logger.log_image_to_video(image=observation)

                # Transform all values
                observation = self.obs_transform(observation)
                observation_queue.append(observation)
                action = self.action_transform(action)
                reward = self.reward_transform(reward)

                # Save sample to replay buffer if queue is full
                if len(observation_queue) == 3:
                    self.replay_buffer.add_sample(
                        observation_1=observation_queue[0],
                        observation_2=observation_queue[1],
                        observation_3=observation_queue[2],
                        action=action,
                        reward=reward,
                        terminated=terminated,
                    )
                else:
                    logger.info("Observation queue is not full, buffering.")

                # Set variables for next iteration
                step_count += 1
                self.total_env_step_count += 1
                episode_reward_sum += reward

            # Record metrics
            episode_expected_reward = episode_reward_sum / step_count
            episode_metrics["expected_reward"] = episode_expected_reward
            episode_metrics["episode_length"] = step_count

            # Save video to disk and clear buffer
            if self.logger.video_option:
                self.logger.save_and_clear_video(video_name=f"episode_{episode_count}")

            # Log metrics
            if episode_count % self.logging_config.episode_log_interval == 0:
                self.logger.log_metrics(episode_metrics, self.total_env_step_count)

            episode_count += 1

    def episode_active(self, step_count, terminated):
        # Inactive if max steps is reached or termination is true
        if self.max_env_steps is not None:
            step_condition = step_count < self.max_env_steps
        else:
            step_condition = True

        return step_condition and not terminated

    def train_models(self):
        """
        Different training losses are:
            - Reconstruction loss
            - Reward prediction loss
            - Transition prediction loss

        Trying all in one training first. If this doesn't work,
        will switch to individual optimizers or separate updates.
        :return:
        """
        logger.info("Training models.")

        # Train models
        for i in range(self.training_steps):
            # Sample a batch from the replay buffer
            batch = self.replay_buffer.sample_batch(self.batch_size)

            # Unpack batch
            obs_1, obs_2, obs_3 = (
                batch["observation_1"],
                batch["observation_2"],
                batch["observation_3"],
            )
            action, reward = batch["action"], batch["reward"]

            # Get all model outputs
            encoder_state_hat = self.models["encoder"](obs_1, obs_2)
            encoder_next_state_hat = self.models["encoder"](obs_2, obs_3)
            obs_1_hat, obs_2_hat_1 = self.models["decoder"](encoder_state_hat)
            obs_2_hat_2, obs_3_hat = self.models["decoder"](encoder_next_state_hat)
            transition_next_state_hat = self.models["transition_model"](
                encoder_state_hat, action
            )
            reward_hat = self.models["reward_model"](encoder_state_hat)

            # Calculate losses
            obs_hat = torch.cat((obs_1_hat, obs_2_hat_1, obs_2_hat_2, obs_3_hat), dim=0)
            obs = torch.cat((obs_1, obs_2, obs_2, obs_3), dim=0)
            reconstruction_loss = F.mse_loss(obs_hat, obs)
            transition_loss = F.mse_loss(
                transition_next_state_hat, encoder_next_state_hat
            )
            reward_loss = F.mse_loss(reward_hat, reward)
            total_loss = (
                self.reconstruction_loss_weight * reconstruction_loss
                + self.transition_loss_weight * transition_loss
                + self.reward_loss_weight * reward_loss
            )

            # Update models
            self.model_optimizer.zero_grad()
            total_loss.backward()
            self.model_optimizer.step()

            # Record metrics
            training_metrics = {
                "reconstruction_loss": reconstruction_loss.item(),
                "transition_loss": transition_loss.item(),
                "reward_loss": reward_loss.item(),
                "total_loss": total_loss.item(),
            }
            if (
                self.total_training_step_count
                % self.logging_config.training_log_interval
                == 0
            ):
                self.logger.log_metrics(
                    training_metrics, self.total_training_step_count
                )

            self.total_training_step_count += 1

        # TODO: Checkpoint models

    def train(self):
        # TODO: Add convergence check
        converged = False
        # while not converged:
        #     self.generate_samples()
        #     self.train_models()
        for i in range(2):
            self.generate_samples()
            self.train_models()

    def eval(self):
        self.generate_samples()

    @classmethod
    def reload_from_ckpt(cls, ckpt_path):
        # TODO: Implement
        raise NotImplementedError

    @classmethod
    def merge_ckpt_config(cls, config, ckpt_config):
        # TODO: Implement
        raise NotImplementedError
