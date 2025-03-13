import torch
import torch.nn as nn
import re

#Llava official code의 projector 그대로 가져옴
#projector 구조 맞는지 잘 확인해보기

class IdentityMap(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x, *args, **kwargs):
    return x
  
  @property
  def config(self):
    return {"nn_projector_type" : 'identity'}

class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

def build_vision_projector(config, input_dim = None, delay_load=False, **kwargs):
    if input_dim is None:
        input_dim = config['mm_hidden_size']

    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(input_dim, config['hidden_size'])
    
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(input_dim, config['hidden_size'])]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config['hidden_size'], config['hidden_size']))
        return nn.Sequential(*modules)
    if projector_type == 'identity':
        return IdentityMap()
    
    raise ValueError(f'Unknown projector type: {projector_type}')