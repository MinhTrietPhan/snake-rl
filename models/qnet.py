import torch
import torch.nn as nn

class QNet(nn.Module):

    def __init__(
        self,
        input_size=28,
        output_size=3,
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, output_size),

        )

    def forward(self, x):
        return self.network(x)
