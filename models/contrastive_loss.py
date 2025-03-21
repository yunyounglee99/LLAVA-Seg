# this is for the option

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
  def __init__(self, temperature = 0.07):
    super(ContrastiveLoss, self).__init__()
    self.temperature = temperature

  def forward(self, image_features, text_features):
    #image_features : (B, D), text_features : (B, D) 

    image_features = F.normalize(image_features, dim = 1)
    text_features = F.normalize(text_features, dim = 1)

    logits = torch.matmul(image_features, text_features.T) / self.temperature

    labels = torch.arange(logits.shape[0]).to(logits.device)

    loss = F.cross_entropy(logits, labels)

    return loss