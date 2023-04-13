import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn


class Decoder(pl.LightningModule):
    """
    model(s) -> o_1, o_2. Deconvolutional model.
    """

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.fc1 = nn.Linear(20, 84)
        self.fc2 = nn.Linear(84, 120)
        self.fc3 = nn.Linear(120, 640)
        self.deconv1 = nn.ConvTranspose2d(
            20,
            16,
            kernel_size=4,
            stride=3,
            output_padding=1,
        )
        self.deconv2 = nn.ConvTranspose2d(
            16,
            6,
            kernel_size=4,
            stride=2,
            output_padding=1,
        )
        self.deconv3 = nn.ConvTranspose2d(
            6,
            1,
            kernel_size=4,
            stride=2,
            padding=0,
            output_padding=0,
        )

    def forward(self, state):
        """
        Feed both observations through the encoder and concatenate the results.
        Then pass through the fully connected layers.
        :param o_1:
        :param o_2:
        :return:
        """
        o = F.relu(self.fc1(state))
        o = F.relu(self.fc2(o))
        o = F.relu(self.fc3(o))
        o_1 = o[:, :320]
        o_1 = torch.reshape(o_1, (o_1.shape[0], 20, 4, 4))
        o_2 = o[:, 320:]
        o_2 = torch.reshape(o_2, (o_2.shape[0], 20, 4, 4))
        o_1 = self.conv_decoder(o_1)
        o_2 = self.conv_decoder(o_2)

        return o_1, o_2

    def conv_decoder(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x
