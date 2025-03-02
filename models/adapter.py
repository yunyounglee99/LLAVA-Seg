import torch
import torch.nn as nn

class SegAdapter(nn.Module):
  def __init__(self, in_channels, num_classes):
    super(SegAdapter, self).__init__()

    self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size = 3, padding = 1)
    self.relu = nn.ReLU(inplace = True)
    self.conv2 = nn.Conv2d(in_channels // 2, num_classes, kernel_size = 1)

    def forward(self, x):

      x = self.conv1(x)
      x = self.relu(x)
      x = self.conv2(x)
      return x