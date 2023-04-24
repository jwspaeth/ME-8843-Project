import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn


class Encoder(pl.LightningModule):
    """
    model(o_1, o_2) -> s. Convolutional model.
    """

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(640, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 30)

    def forward(self, o_1, o_2):
        """
        Feed both observations through the encoder and concatenate the results.
        Then pass through the fully connected layers.
        :param o_1:
        :param o_2:
        :return:
        """
        s_1 = self.conv_encoder(o_1)
        s_1 = torch.flatten(s_1, 1)  # flatten all dimensions except batch
        s_2 = self.conv_encoder(o_2)
        s_2 = torch.flatten(s_2, 1)
        s = torch.cat((s_1, s_2), 1)
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = torch.sigmoid(self.fc3(s))
        return s

    def conv_encoder(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x
