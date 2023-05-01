import logging
import os

import cv2
import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from me_8843_project.core.ObservationQueue import ObservationQueue
from me_8843_project.core.ReplayBuffer import ReplayBuffer
from me_8843_project.core.WeightsAndBiasesWriter import WeightsAndBiasesWriter
from omegaconf import DictConfig, OmegaConf, read_write
from tqdm import tqdm

logger = logging.getLogger(__name__)

"""
- Add actions to reward model
- What to do about episode termination spike?
"""


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
        max_training_epochs=1,
        ckpt_config=None,
        checkpoint_interval=-1,
        checkpoint_folder="data/checkpoints",
        num_checkpoints=10,
        obs_size=None,
        obs_gray_conversion=False,
        reconstruction_lr=1e-3,
        reward_lr=1e-3,
        transition_lr=1e-3,
        batch_size=32,
        total_lr=1e-3,
    ):
        # Setup environment
        self.env = gym.make(**env_config)

        # Setup models and place on GPU if available
        self.models = {
            "encoder": hydra.utils.instantiate(encoder_config),
            "decoder": hydra.utils.instantiate(decoder_config),
            "transition": hydra.utils.instantiate(transition_model_config),
            "reward": hydra.utils.instantiate(reward_model_config),
        }

        # Print number of model parameters
        for model_name, model in self.models.items():
            logger.info(
                f"{model_name}: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters"
            )

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
        self.max_training_epochs = max_training_epochs
        self.eval_config = eval_config
        self.ckpt_config = ckpt_config
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_folder = checkpoint_folder
        self.num_checkpoints = num_checkpoints
        self.obs_size = obs_size
        self.obs_gray_conversion = obs_gray_conversion
        self.batch_size = batch_size
        self.reconstruction_lr = reconstruction_lr
        self.reward_lr = reward_lr
        self.transition_lr = transition_lr
        self.total_lr = total_lr

        # Setup model optimizer
        self.model_optimizers = self.create_model_optimizers()

        # Setup counters
        self.total_env_step_count = 0
        self.total_training_step_count = 0

    def obs_transform(self, obs):
        if self.obs_size is not None:
            obs = cv2.resize(obs, self.obs_size)

        if self.obs_gray_conversion:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = np.expand_dims(obs, axis=2)
            obs = obs / 255.0  # Normalize

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
        logger.info("\tGenerating samples.")
        logger.info("\tPolicy: {}".format(self.policy.__class__.__name__))

        # Continue training until max_episodes or convergence is reached
        # Queue holds most recent 3 observations, which is required for training
        observation_queue = ObservationQueue(maxlen=3)
        episode_count = 0
        full_metrics = {
            "total_expected_reward": None,
        }
        episode_metrics = {
            "expected_reward": None,
            "episode_length": None,
        }
        while episode_count < self.max_env_episodes:
            logger.info(f"\t\tEpisode: {episode_count}")
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

                    # Save reconstructed frame
                    if self.logger.reconstruction_option:
                        with torch.no_grad():
                            # Get reconstructed frame
                            state_hat = self.models["encoder"](
                                observation_queue[-2], observation_queue[-1]
                            )
                            reconstruction = self.models["decoder"](state_hat)[1]
                            reconstruction = reconstruction.cpu().numpy() * 255
                            reconstruction = reconstruction.astype(np.uint8)
                            reconstruction = np.squeeze(reconstruction, axis=0)
                            reconstruction = np.transpose(reconstruction, (1, 2, 0))
                        original = observation_queue[-1].cpu().numpy() * 255
                        original = original.astype(np.uint8)
                        original = np.squeeze(original, axis=0)
                        original = np.transpose(original, (1, 2, 0))
                        joined = np.concatenate((original, reconstruction), axis=1)
                        self.logger.log_recon_to_video(image=joined)
                else:
                    logger.info("\t\tObservation queue is not full, buffering.")

                # Set variables for next iteration
                step_count += 1
                self.total_env_step_count += 1
                episode_reward_sum += reward

            # Record metrics
            episode_expected_reward = episode_reward_sum / step_count
            episode_metrics["expected_reward"] = episode_expected_reward
            episode_metrics["episode_length"] = step_count
            episode_metrics["replay_buffer_size"] = len(self.replay_buffer)

            # Save video to disk and clear buffer
            if self.logger.video_option:
                self.logger.save_and_clear_video(video_name=f"episode_{episode_count}")

            # Save reconstruction to disk and clear buffer
            if self.logger.reconstruction_option:
                self.logger.save_and_clear_recon(
                    video_name=f"recon_episode_{episode_count}"
                )

            # Log metrics
            if full_metrics["total_expected_reward"] is None:
                full_metrics["total_expected_reward"] = episode_expected_reward.item()
            else:
                full_metrics["total_expected_reward"] += episode_expected_reward.item()

            episode_count += 1

        full_metrics["total_expected_reward"] /= episode_count
        logger.info(
            f"\t\tTotal expected reward: {full_metrics['total_expected_reward']}"
        )

    def episode_active(self, step_count, terminated):
        # Inactive if max steps is reached or termination is true
        if self.max_env_steps is not None:
            step_condition = step_count < self.max_env_steps
        else:
            step_condition = True

        return step_condition and not terminated

    def create_model_optimizers(self):
        model_optimizers = {
            "reconstruction": None,
            "reward": None,
            "transition": None,
        }

        # Reconstruction optimizers optimizes both encoder and decoder
        reconstruction_params = list(self.models["encoder"].parameters()) + list(
            self.models["decoder"].parameters()
        )
        model_optimizers["reconstruction"] = torch.optim.Adam(
            reconstruction_params, lr=self.reconstruction_lr
        )

        reward_params = list(self.models["reward"].parameters())
        model_optimizers["reward"] = torch.optim.Adam(reward_params, lr=self.reward_lr)

        transition_params = list(self.models["transition"].parameters())
        model_optimizers["transition"] = torch.optim.Adam(
            transition_params, lr=self.transition_lr
        )

        # reconstruction_params = list(self.models["decoder"].parameters())
        # model_optimizers["reconstruction"] = torch.optim.Adam(
        #     reconstruction_params, lr=self.reconstruction_lr
        # )
        #
        # reward_params = list(self.models["encoder"].parameters()) + list(
        #     self.models["reward"].parameters()
        # )
        # model_optimizers["reward"] = torch.optim.Adam(reward_params, lr=self.reward_lr)
        #
        # transition_params = list(self.models["transition"].parameters())
        # model_optimizers["transition"] = torch.optim.Adam(
        #     transition_params, lr=self.transition_lr
        # )

        return model_optimizers

    def checkpoint_models(self):
        # Create checkpoint folder if it doesn't exist
        os.makedirs(self.checkpoint_folder, exist_ok=True)

        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(self.checkpoint_folder, f"{model_name}.pt")
            torch.save(model.state_dict(), model_path)

    def reload_models(self, checkpoint_folder):
        # Load models
        for model_name, model in self.models.items():
            model_path = os.path.join(checkpoint_folder, f"{model_name}.pt")
            model.load_state_dict(torch.load(model_path))

    def compute_loss(self, batch):
        obs_1, obs_2, obs_3 = (
            batch["observation_1"],
            batch["observation_2"],
            batch["observation_3"],
        )
        action, reward = batch["action"], batch["reward"]

        # Get all model outputs
        encoder_state_hat = self.models["encoder"](obs_1, obs_2)
        encoder_next_state_hat = self.models["encoder"](obs_2, obs_3)
        obs_1_hat, obs_2_hat_1 = self.models["decoder"](
            encoder_state_hat
        )  # Need to maintain gradient with encoder
        obs_2_hat_2, obs_3_hat = self.models["decoder"](encoder_next_state_hat)
        transition_next_state_hat = self.models["transition"](
            encoder_state_hat.detach(), action
        )
        reward_hat = self.models["reward"](encoder_state_hat.detach())

        # Calculate losses
        losses = {}
        obs_hat = torch.cat((obs_1_hat, obs_2_hat_1, obs_2_hat_2, obs_3_hat), dim=0)
        obs = torch.cat((obs_1, obs_2, obs_2, obs_3), dim=0)
        losses["reconstruction"] = F.mse_loss(obs_hat, obs)
        losses["transition"] = F.mse_loss(
            transition_next_state_hat, encoder_next_state_hat.detach()
        )
        losses["reward"] = F.mse_loss(reward_hat, reward)

        metrics = {}
        reward = reward.cpu().numpy().transpose()
        reward_hat = reward_hat.cpu().detach().numpy().transpose()
        metrics["reward_correlation"] = torch.Tensor(
            [np.corrcoef(reward, reward_hat)[0, 1]]
        )

        return losses, metrics

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
        logger.info(
            f"\tTraining models. Replay buffer length: {len(self.replay_buffer)}"
        )
        dataloader = self.replay_buffer.get_dataloader(self.batch_size)

        # Train models
        for i, batch in enumerate(tqdm(dataloader)):
            losses, metrics = self.compute_loss(batch)

            # Record metrics
            if i == 0:
                # Initialize metrics
                training_step_metrics = {}
                for key in losses.keys():
                    training_step_metrics[key + "_loss"] = losses[key].item()

                for key in metrics.keys():
                    training_step_metrics[key] = metrics[key].item()
            else:
                # Update metrics
                for key in losses.keys():
                    training_step_metrics[key + "_loss"] += losses[key].item()

                for key in metrics.keys():
                    training_step_metrics[key] += metrics[key].item()

            # Update models
            for key in self.model_optimizers.keys():
                self.model_optimizers[key].zero_grad()
                losses[key].backward()
                self.model_optimizers[key].step()

            self.total_training_step_count += 1

        # Average metrics
        for key in losses.keys():
            training_step_metrics[key + "_loss"] = training_step_metrics[
                key + "_loss"
            ] / len(dataloader)

        for key in metrics.keys():
            training_step_metrics[key] = training_step_metrics[key] / len(dataloader)

        # Log metrics
        self.logger.log_metrics(training_step_metrics, self.total_training_step_count)

        self.checkpoint_models()

    def train(self):
        total_training_epochs = 0
        while not self.train_end_condition(total_training_epochs):
            logger.info(f"Epoch: {total_training_epochs}")
            self.generate_samples()
            self.train_models()
            total_training_epochs += 1
        logger.info(f"Training complete.")

    def train_end_condition(self, total_training_epochs):
        if self.max_training_epochs is None:
            return True  # TODO: Use better check here
        else:
            return total_training_epochs >= self.max_training_epochs

    def eval(self):
        self.generate_samples()

    @classmethod
    def reload_from_ckpt(cls, config):
        # Load config from checkpoint
        ckpt_folder_path = config.reload_config.path
        config_path = os.path.join(ckpt_folder_path, "config.yaml")
        with open(config_path, "r") as f:
            ckpt_config = OmegaConf.load(f)

        # Merge configs, preserving relevant current config values
        # config = cls.merge_ckpt_config(config, ckpt_config)

        # Create trainer
        trainer = cls(**config.trainer_config)

        # Reload models
        trainer.reload_models(ckpt_folder_path)

        return trainer

    @classmethod
    def merge_ckpt_config(cls, config, ckpt_config):
        # Choose values to merge from ckpt_config
        with read_write(config):
            config.trainer_config.eval_config = cls.config_copy(
                config.trainer_config.eval_config,
                ckpt_config.trainer_config.eval_config,
            )
            config.trainer_config.logging_config = cls.config_copy(
                config.trainer_config.logging_config,
                ckpt_config.trainer_config.logging_config,
            )
            config.trainer_config.batch_size = ckpt_config.trainer_config.batch_size

        return config

    @classmethod
    def config_copy(cls, config_1, config_2):
        # Copy config_2 into config_1
        for key, value in config_2.items():
            if isinstance(value, dict) or isinstance(value, DictConfig):
                cls.config_copy(config_1[key], value)
            else:
                config_1[key] = value

        return config_1
