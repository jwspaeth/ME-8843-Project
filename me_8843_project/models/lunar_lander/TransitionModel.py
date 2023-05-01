import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn


class TransitionModel(pl.LightningModule):
    """
    model(s, a) -> s'. Fully connected model.
    """

    def __init__(self, state_dim=40, action_dim=3):
        super().__init__()
        self.save_hyperparameters()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim + self.action_dim, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)

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
        r = torch.sigmoid(self.fc3(r))

        return r
