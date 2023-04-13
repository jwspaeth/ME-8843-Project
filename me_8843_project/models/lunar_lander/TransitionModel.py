import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn


class TransitionModel(pl.LightningModule):
    """
    model(s, a) -> s'. Fully connected model.
    """

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.state_dim = 20
        self.action_dim = 2

        self.fc1 = nn.Linear(22, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 20)

    def forward(self, state, action):
        """
        Feed both observations through the encoder and concatenate the results.
        Then pass through the fully connected layers.
        :param o_1:
        :param o_2:
        :return:
        """
        x = torch.cat((state, action), 1)
        r = F.relu(self.fc1(x))
        r = F.relu(self.fc2(r))
        r = F.relu(self.fc3(r))

        return r
