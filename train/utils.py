import torch

def save_checkpoint(model, optimizer, epoch, path):
  torch.save({
    "epoch" : epoch,
    "model_state_dict" : model.state_dict(),
    "optimzier_state_dict" : optimizer.state_dict()
  }, path)

def load_checkpoint(model, optimizer, path, device = "cuda"):
  checkpoint =  torch.load(path, map_location = device)
  model.load_state_dict(checkpoint["model_state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
  return checkpoint["epoch"]