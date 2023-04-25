import copy

import torch


class ObservationQueue(list):
    """
    FIFO queue for storing observations. The queue is limited to a maximum length.
    """

    def __init__(self, maxlen=3):
        super().__init__()
        self.maxlen = maxlen

    def append(self, observation):
        if torch.cuda.is_available():
            observation = observation.cuda()
        super().append(copy.deepcopy(observation))
        if len(self) > self.maxlen:
            self.pop(0)
