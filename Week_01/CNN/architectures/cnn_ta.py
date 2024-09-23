import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)  # Updated dimensions: 64 * 16 * 16
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Reorder from [batch_size, height, width, channels] to [batch_size, channels, height, width]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)  # Use reshape instead of view
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x