import torch.nn as nn
import torch
from transformers import CLIPVisionModel
from .adapter import SegAdapter
from .unet_adapter import UNetAdapter

class SegEncoder(nn.Module):
  def __init__(self, num_seg_classes, adapter_type = 'cnn', device = "cuda", fuse = True):
    super(SegEncoder, self).__init__()

    self.base_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
    self.device = device
    self.fuse = fuse
    self.num_seg_classes = num_seg_classes
    
    hidden_size = self.base_encoder.config.hidden_size
    if adapter_type == 'cnn':
      self.adapter = SegAdapter(in_channels = hidden_size, num_classes = num_seg_classes)
    elif adapter_type == 'unet':
      self.adapter = UNetAdapter(in_channels = hidden_size, num_classes = num_seg_classes)
    else:
      raise ValueError(f"Unknown adapter type : {adapter_type}")
    
    #fusion feature : hdden_size + num_seg_classes
    if self.fuse:
      self.fusion_proj = nn.Linear(hidden_size + num_seg_classes, hidden_size) # 기존의 projector 사용가능한지 더 생각해보기

  def forward(self, pixel_values):
    outputs = self.base_encoder(pixel_values = pixel_values)

    B, N, C = outputs.last_hidden_state.shape

    H = W = int(N**0.5)
    features = outputs.last_hidden_state.transpose(1,2).reshape(B, C, H, W)
    seg_logits = self.adapter(features)

    global_feature = features.mean(dim = [2,3])

    if self.fuse:
      seg_probs = torch.softmax(seg_logits, dim = 1)
      seg_embeds = seg_probs.mean(dim=[2,3])
      fused_feature = torch.cat([global_feature, seg_embeds], dim = 1)
      visual_embeds = fused_feature
    else:
      visual_embeds = global_feature

    return {"visual_embeds" : visual_embeds, "seg_logits" : seg_logits}