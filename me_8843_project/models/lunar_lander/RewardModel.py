import lightning.pytorch as pl
import torch.nn.functional as F
from torch import nn


class RewardModel(pl.LightningModule):
    """
    model(s) -> r. Fully connected model.
    """

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.fc1 = nn.Linear(500, 250)
        self.fc2 = nn.Linear(250, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, state):
        """
        Feed both observations through the encoder and concatenate the results.
        Then pass through the fully connected layers.
        :param o_1:
        :param o_2:
        :return:
        """
        r = F.relu(self.fc1(state))
        r = F.relu(self.fc2(r))
        r = self.fc3(r)

        return r
