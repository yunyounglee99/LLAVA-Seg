import torch
import torch.nn as nn

class SegAdapter(nn.Module):
  def __init__(self, in_channels, num_classes):
    super(SegAdapter, self).__init__()
    self.conv_block = nn.Sequential(
      nn.Conv2d(in_channels, in_channels // 2, kernel_size = 3, padding = 1),
      nn.ReLU(inplace = True),
      nn.Conv2d(in_channels // 2, num_classes, kernel_size = 1)
    )

  def forward(self, x):
    return self.conv_block(x)