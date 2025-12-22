from torch import nn

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim, n_units = 10):
        self.net = nn.Sequential(
            nn.Linear(input_dim, n_units),
            nn.Linear(n_units, n_units),
            nn.Linear(n_units, output_dim)
        )
    def forward(self, x):
        return self.net(x)