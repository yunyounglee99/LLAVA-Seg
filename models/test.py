from models.llava_model import LLaVAModel
from train.train import load_config
from models.vision_encoder import SegEncoder
from transformers import CLIPVisionModel

model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
print(model.config.hidden_size)

config = load_config('/Users/nyoung/Desktop/dev/project/LLAVA-Seg/configs/pretrain.yaml')
model2 = LLaVAModel(10, config)

print(model2)