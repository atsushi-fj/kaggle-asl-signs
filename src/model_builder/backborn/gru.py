from torch import nn
from torchinfo import summary


class GRU(nn.Module):
    def __init__(self,
                 input_size):
        super().__init__()
        self.gru1 = nn.GRU(input_size, 64,
                           dropout=0.1,
                           batch_first=True)
        self.gru2 = nn.GRU(64, 128,
                           dropout=0.1,
                           batch_first=True)
        self.gru3 = nn.GRU(128, 64,
                           dropout=0.1,
                           batch_first=True)
        self.flatten = nn.Flatten(1, -1)
        self.dense1 = nn.Linear(543*64, 64)
        self.dense2 = nn.Linear(64, 32)
        self.dense3 = nn.Linear(32, 250)
        self.relu = nn.ReLU()
    
    
    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.relu(x)
        x, _ = self.gru2(x)
        x = self.relu(x)
        x, _ = self.gru3(x)
        x = self.flatten(self.relu(x))
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        return x


if __name__ == "__main__":
    model = GRU(3)
    summary(model, 
            input_size=(1, 543, 3),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

