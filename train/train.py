# this is for training vision encoder
import torch
from torch.utils.data import DataLoader
from dataset.coco_stuff import CocoStuffDataset
from models.llava_model import LLaVAModel
import yaml
import argparse

def load_config(config_path):
  with open(config_path, "r") as f:
    return yaml.safe_load(f)
  
def train_pretrain(config):
  device = "cuda" if torch.cuda.is_availabele() else "cpu"
  dataset = CocoStuffDataset(root = config["data_path"], split = "train")
  dataloader = DataLoader(dataset, batch_size = config["batch_size"], shuffle=True, num_workers = 4)

  model = LLaVAModel(num_seg_classes = config["num_classes"], config = config, device = device)
  model.to(device)

  optimizer = torch.optim.Adam(model.vision_encoder.parameters(), lr = config["learning_rate"])
  seg_loss_fn = torch.nn.CrossEntropyLoss()

  model.train()
  for epoch in range(config["num_epochs"]):
    for batch in dataloader:
      images = batch["image"].to(device)
      masks = batch["mask"].to(device).long
      optimizer.zero_grad()
      out = model.vision_encoder(images)
      seg_logits = out["seg_logits"]
      loss = seg_loss_fn(seg_logits, masks)
      loss.backward()
      optimizer.step()
    print(f"Epoch {epoch+1}/{config['num_epochs']} - Segmentation Loss : {loss.item():.4f}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config,", type = str, default = "configs/pretrain_cocostuff.yaml")
  args = parser.parse_args()
  config = load_config(args.config)
  train_pretrain(config)
