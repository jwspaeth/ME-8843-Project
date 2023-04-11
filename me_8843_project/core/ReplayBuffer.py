import random
from dataclasses import dataclass


@dataclass
class Sample:
    """
    Sample class to store a single experience.
    """

    observation_1: object
    reward: float
    action: object
    observation_2: object
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
        if self.max_length is not None:
            self.buffer = self.buffer[: self.max_length]

    def add_sample(self, observation_1, reward, action, observation_2, terminated):
        """
        Add an observation to the replay buffer. Assume list is in temporal order.
        """
        sample = Sample(
            observation_1=observation_1,
            reward=reward,
            action=action,
            observation_2=observation_2,
            terminated=terminated,
        )

        self.buffer.insert(0, sample)
        self.truncate_buffer()

    def sample_batch(self, batch_size):
        """
        Sample a batch of experiences from the replay buffer.
        """
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        """
        Return the size of the replay buffer.
        """
        return len(self.buffer)
