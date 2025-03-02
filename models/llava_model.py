import torch
import torch.nn as nn
from transformers import LlavaForConditionalGeneration

from .vision_encoder import SegEncoder
from .projector import build_vision_projector

class LLaVAModel(nn.Module):
  def __init__(self, num_seg_classes, config, device = "cuda"):
    super(LLaVAModel, self).__init__()
    self.device = device

    self.vision_encoder = SegEncoder(num_seg_classes = num_seg_classes, device = device)

    self.language_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")

    self.image_proj = build_vision_projector(config)

  def forward(self, pixel_values, input_ids = None, attention_mask = None, labels = None):
    vision_out = self.vision_encoder(pixel_values)
    visual_embeds = vision_out["visual_embeds"]
    seg_logits = vision_out["seg_logits"]
    projected_img = self.image_proj(visual_embeds)

    if input_ids is not None:
      inputs_embeds = self.language_model.model.embed_tokens(input_ids)
      inputs_embeds = torch.cat([projected_img.unsqueeze(1), inputs_embeds], dim=1)
    else:
      inputs_embeds = projected_img.unsqueeze(1)
    lm_outputs = self.language_model(
      inputs_embeds = inputs_embeds,
      attention_mask = attention_mask,
      labels = labels
    )

    return {"lm_loss" : lm_outputs.loss, "logits" : lm_outputs.logits, "seg_logits" : seg_logits}