import torch.nn as nn
from torchinfo import summary


class GRU(nn.Module):
    def __init__(self,
                 input_size=[32, 543, 3],
                 dropout_rate=0.1):
        super().__init__()
        self.input_size = input_size
        self.gru = nn.GRU(self.input_size[-1], 64, 3,
                           dropout=dropout_rate,
                           batch_first=True)
        self.dense1 = nn.Linear(64, 64)
        self.dense2 = nn.Linear(64, 32)
        self.dense3 = nn.Linear(32, 250)
        self.relu = nn.ReLU()
    
    
    def forward(self, x):
        x, _ = self.gru(x)
        x = self.relu(self.dense1(x[:, -1, :]))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        return x


if __name__ == "__main__":
    model = GRU()
    summary(model, 
            input_size=(1, 543, 3),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

