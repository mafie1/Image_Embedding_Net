import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision.transforms as T
from PIL import Image


def train_transform(IMAGE_HEIGHT, IMAGE_WIDTH):
    transform = A.Compose([
        #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        #A.RandomCrop(height= IMAGE_HEIGHT, width= IMAGE_WIDTH, ),
        #A.Rotate(limit=35, p=1.0),
        #A.HorizontalFlip(p=0.5),
        #A.VerticalFlip(p=0.1),
        #A.Normalize(
         #   mean=[0.0, 0.0, 0.0],
          #  std=[1.0, 1.0, 1.0],
           # max_pixel_value = 255.0, ),
        ToTensorV2(),
        ])
    return transform


def mask_train_transform(IMAGE_HEIGHT, IMAGE_WIDTH):
    transform = T.Compose(
        [T.ToPILImage(),
         T.Resize((IMAGE_HEIGHT,IMAGE_WIDTH), interpolation = Image.NEAREST),
         T.ToTensor()])
    return transform


def image_train_transform(IMAGE_HEIGHT, IMAGE_WIDTH):
    transform = T.Compose(
        [T.ToPILImage(),
         T.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
         T.ToTensor(),
         #T.Normalize(
          #   mean=[0.485, 0.456, 0.406],
           #  std=[0.229, 0.224, 0.225])
         ])
    return transform

