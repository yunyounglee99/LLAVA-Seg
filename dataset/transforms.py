from torchvision import transforms

def get_transforms():
  def transform_fn(image, mask): 
    image_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean = [0,485, 0,456, 0.406],
                          std = [0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
      transforms.Resize((224, 224))
    ])
    image = image_transform(image)
    mask = mask_transform(mask)
    return image, mask
  
  return transform_fn