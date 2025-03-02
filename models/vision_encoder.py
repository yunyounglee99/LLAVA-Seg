import torch.nn as nn
import torch
from transformers import CLIPVisionModel
from .adapter import SegAdapter

class SegEncoder(nn.Module):
  def __init__(self, num_seg_classes, device = "cuda"):
    super(SegEncoder, self).__init__()

    self.base_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    self.device = device
    
    hidden_size = self.base_encder.config.hidden_size
    self.adapter = SegAdapter(in_channels = hidden_size, num_classes = num_seg_classes)

  def forward(self, pixel_values):
    outputs = self.base_encoder(pixel_values = pixel_values)

    B, N, C = outputs.last_hidden_state.shape

    H = W = int(N**0.5)
    features = outputs.last_hidden_state.traspose(1,2).reshape(B, C, H, W)
    seg_logits = self.adapter(features)

    visual_embeds = features.mean(dim = [2,3])

    return {"visual_embeds" : visual_embeds, "seg_logits" : seg_logits}