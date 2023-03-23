import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary


class GRU(nn.Module):
    def __init__(self,
                 input_size=3,
                 dropout_rate=0.1,
                 n_layers=3):
        super().__init__()
        self.gru = nn.GRU(input_size, 560, n_layers,
                           dropout=dropout_rate,
                           batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(560, 560),
            nn.ReLU()
        )
        self.dense1 = self.dense * (n_layers-2)
        self.dense2 = nn.Sequential(
            nn.Linear(560, 280),
            nn.ReLU(),
            nn.Linear(280, 250)
        )
    
    
    def forward(self, x):
        x, _ = self.gru(x)
        x = self.dense1(x[:, -1, :])
        x = self.dense2(x)
        return x


if __name__ == "__main__":
    model = GRU()
    summary(model, 
            input_size=(1, 543, 3),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

