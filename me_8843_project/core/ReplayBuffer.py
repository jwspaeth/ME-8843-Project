import random
from dataclasses import dataclass

import torch


@dataclass
class Sample:
    """
    Sample class to store a single experience.
    """

    observation_1: object
    observation_2: object
    observation_3: object
    reward: float
    action: object
    terminated: bool


class ReplayBuffer:
    """
    Replay Buffer for storing experiences that the agent can then use for training.
    Organized in list format.
    Inefficient memory storage because it stores observations twice.
    Indicated as true states with the true_state flag.
    """

    def __init__(self, max_length=None, true_states=True):
        self.buffer = []
        self.max_length = max_length
        self.true_states = true_states

    def truncate_buffer(self):
        """
        Truncate the replay buffer to the max length.
        """
        if self.max_length is not None and len(self.buffer) > self.max_length:
            self.buffer = self.buffer[: self.max_length]

    def add_sample(
        self, observation_1, observation_2, observation_3, reward, action, terminated
    ):
        """
        Add an observation to the replay buffer. Assume list is in temporal order.
        """
        sample = Sample(
            observation_1=observation_1,
            observation_2=observation_2,
            observation_3=observation_3,
            reward=reward,
            action=action,
            terminated=terminated,
        )

        self.buffer.append(sample)
        self.truncate_buffer()

    def sample_batch(self, batch_size):
        """
        Sample a batch of experiences from the replay buffer.
        """
        batch = random.sample(self.buffer, batch_size)
        batch_dict = {
            "observation_1": [sample.observation_1 for sample in batch],
            "observation_2": [sample.observation_2 for sample in batch],
            "observation_3": [sample.observation_3 for sample in batch],
            "reward": [sample.reward for sample in batch],
            "action": [sample.action for sample in batch],
            "terminated": [sample.terminated for sample in batch],
        }
        batch_dict = {
            "observation_1": torch.cat(batch_dict["observation_1"], dim=0),
            "observation_2": torch.cat(batch_dict["observation_2"], dim=0),
            "observation_3": torch.cat(batch_dict["observation_3"], dim=0),
            "reward": torch.cat(batch_dict["reward"], dim=0),
            "action": torch.cat(batch_dict["action"], dim=0),
        }
        return batch_dict

    def __len__(self):
        """
        Return the size of the replay buffer.
        """
        return len(self.buffer)
