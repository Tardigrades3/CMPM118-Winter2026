import torch 
from torch import nn

class NinaProClassifyCNNSmall(nn.Module):
  def __init__(self, final_layer_size):
    super().__init__()
    self.conv1 = nn.Conv1d(10, 128, kernel_size=5, stride=2)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool1d(kernel_size=2)
    self.drop1 = nn.Dropout(0.1)

    self.flatten = nn.Flatten()
    
    self.dense1 = nn.LazyLinear(256)
    self.relu3 = nn.ReLU()
    self.drop3 = nn.Dropout(0.4)
    self.finalLayer = nn.Linear(256, final_layer_size)

  def forward(self, x):
    out = self.relu1(self.conv1(x))
    out = self.pool1(out)
    out = self.drop1(out)
    out = self.flatten(out)
    out = self.relu3(self.dense1(out))
    out = self.drop3(out)
    out = self.finalLayer(out)
    return out

class NinaProClassifyCNN(nn.Module):
  def __init__(self, final_layer_size):
    super().__init__()
    self.conv1 = nn.Conv1d(10, 64, kernel_size=5, stride=2)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool1d(kernel_size=2)
    self.drop1 = nn.Dropout(0.1)

    self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2)
    self.relu2 = nn.ReLU()
    self.pool2 = nn.MaxPool1d(kernel_size=2)
    self.drop2 = nn.Dropout(0.1)

    self.flatten = nn.Flatten()
    
    self.dense1 = nn.LazyLinear(256)
    self.relu3 = nn.ReLU()
    self.drop3 = nn.Dropout(0.4)
    self.dense2 = nn.Linear(256, 128)
    self.relu4 = nn.ReLU()
    self.drop4 = nn.Dropout(0.3)
    self.finalLayer = nn.Linear(128, final_layer_size)

  def forward(self, x):
    out = self.relu1(self.conv1(x))
    out = self.pool1(out)
    out = self.drop1(out)
    out = self.relu2(self.conv2(out))
    out = self.pool2(out)
    out = self.drop2(out)
    out = self.flatten(out)
    out = self.relu3(self.dense1(out))
    out = self.drop3(out)
    out = self.relu4(self.dense2(out))
    out = self.drop4(out)
    out = self.finalLayer(out)
    return out