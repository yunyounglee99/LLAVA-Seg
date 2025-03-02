import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, CLIPVisionModel

from .vision_encoder import SegEncoder