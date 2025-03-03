import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetAdapter(nn.Module):
  def __init__(self, in_channels, num_classes, features = [64, 128]):
    super(UNetAdapter, self).__init__()
    # Encoder
    self. enc1 = nn.Sequential(
      nn.Conv2d(in_channels, features[0], kernel_size = 3, padding = 1),
      nn.ReLU(inplace = True),
      nn.Conv2d(features[0], features[0], kernel_size = 3, padding = 1),
      nn.ReLU(inplace = True)
    )
    self.pool1 = nn.MaxPool2d(2)

    self.enc2 = nn.Sequential(
      nn.Conv2d(in_channels, features[1], kernel_size = 3, padding = 1),
      nn.ReLU(inplace = True),
      nn.Conv2d(features[0], features[1], kernel_size = 3, padding = 1),
      nn.ReLU(inplace = True)
    )
    self.pool2 = nn.MaxPool2d(2)

    # Bottleneck
    self.bottleneck = nn.Sequential(
      nn.Conv2d(features[1], features[1]*2, kernel_size = 3, padding = 1),
      nn.ReLU(inplace = True),
      nn.Conv2d(features[1]*2, features[1]*2, kernel_size = 3, padding = 1),
      nn.ReLU(inplace = True)
    )

    # Decoder
    self.upconv2 = nn.ConvTranspose2d(features[1]*2, features[1], kernel_size=2, stride = 2)
    self.dc2 = nn.Sequential(
      nn.Conv2d(features[1]*2, features[1], kernel_size = 3, padding = 1),
      nn.ReLU(inplace = True),
      nn.Conv2d(features[1], features[1], kernel_size = 3, padding = 1),
      nn.ReLU(inplcae = True)
    )

    self.upcon1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
    self.dec1 = nn.Sequential(
        nn.Conv2d(features[0]*2, features[0], kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(features[0], features[0], kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

    self.conv_final = nn.Conv2d(features[0], num_classes, kernel_size = 1)

    def forward(self, x):
      enc1 = self.enc1(x)                   
      enc2 = self.enc2(self.pool1(enc1))     
      bottleneck = self.bottleneck(self.pool2(enc2))  
      
      dec2 = self.upconv2(bottleneck)        
      dec2 = torch.cat([dec2, enc2], dim=1)  
      dec2 = self.dec2(dec2)                 
      
      dec1 = self.upconv1(dec2)               
      dec1 = torch.cat([dec1, enc1], dim=1)   
      dec1 = self.dec1(dec1)                 
      
      out = self.conv_final(dec1)    
              
      return out