import os
from PIL import Image
from torch.utils.data import Dataset
from .transforms import get_transforms

class CocoStuffDataset(Dataset):
  def __init__(self, root, split = "train", transform = None):
    self.root = root
    self.split = split
    self.transform = transform or get_transforms()
    self.image_dir = os.path.join(root, "images", split)
    self.mask_dir = os.path.join(root, "masks", split)
    self.image_files = sorted(os.listdir(self.image_dir))

  def __len__(self):
    return len(self.image_files)
  
  def __getitem__(self, idx):
    image_path = os.path.join(self.image_dir, self.image_files[idx])
    mask_path = os.path.join(self.mask_dir, self.image_files[idx].replace(".jpg", ".png"))
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)
    if self.transform:
      image, mask = self.transform(image, mask)

    return {"image" : image, "mask" : mask}