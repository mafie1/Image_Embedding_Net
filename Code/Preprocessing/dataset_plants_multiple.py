import os
import torch
from PIL import Image
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as T
#from torchvision.transforms import InterpolationMode
#import cv2


def mask_train_transform(IMAGE_HEIGHT, IMAGE_WIDTH):
    transform = T.Compose(
        [T.ToPILImage(),
         T.Resize((IMAGE_HEIGHT,IMAGE_WIDTH), interpolation=0), #InterpolationMode.NEAREST
         T.RandomHorizontalFlip(),
         T.RandomVerticalFlip(),
         T.ToTensor()])
    return transform

def image_train_transform(IMAGE_HEIGHT, IMAGE_WIDTH):
    transform = T.Compose(
        [T.ToPILImage(),
         T.Resize((IMAGE_HEIGHT,IMAGE_WIDTH), interpolation=2), #nterpolationMode.BILINEAR
         T.RandomHorizontalFlip(),
         T.RandomVerticalFlip(),
         T.ToTensor(),
         ])
    return transform

def extended_transform():
    transform = T.Compose(
        [T.RandomHorizontalFlip(),
         T.RandomVerticalFlip(),
         T.RandomCrop(),
         T.ToTensor,
        ]
    )

class CustomDatasetMultiple(Dataset):
    def __init__(self, dir, transform, image_transform, mask_transform):
        self.directory = dir
        self.transform = transform

        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.contents = os.listdir(dir)
        self.images = list(filter(lambda k: 'rgb' in k, self.contents))

        self.store_masks = []
        self.store_images = []

        for index, img in enumerate(self.images):
            img_path = os.path.join(self.directory, self.images[index])
            mask_path = os.path.join(self.directory, self.images[index].replace('rgb.png', 'label.png'))

            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

            uniques = np.unique(mask)
            integers = np.linspace(0, len(uniques) - 1, len(uniques))

            for i in range(0, len(uniques)):
                mask = np.where(mask == uniques[i], integers[i], mask)

            #image = cv2.resize(image, dsize = (512, 512), interpolation=cv2.INTER_LINEAR)
            #mask = cv2.resize(mask, dsize = (512, 512), interpolation=cv2.INTER_NEAREST)

            self.store_masks.append(mask)
            self.store_images.append(image)

        print('Done Initializing Dataset')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.store_images[index]
        mask = self.store_masks[index]

        seed = np.random.randint(np.iinfo('int32').max)

        if self.transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)

            random.seed(seed)
            torch.manual_seed(seed)

            mask = self.transform(mask)

        if self.image_transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)

            image = self.image_transform(image)

        if self.mask_transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)

            mask = self.mask_transform(mask)

        return image, mask


def test():
    img_path = '/Users/luisa/Documents/BA_Thesis/CVPPP2017_instances/training/A1'
    # mask_path = '/Users/luisa/Documents/BA_Thesis/Datasets for Multiple Instance Seg/CVPPP2017_instances/training/A1/plant001_label.png'

    HEIGHT = 512
    WIDTH = 512

    Plants = CustomDatasetMultiple(dir=img_path,
                                   transform=None,
                                   image_transform= image_train_transform(IMAGE_HEIGHT = HEIGHT, IMAGE_WIDTH=WIDTH),
                                   mask_transform= mask_train_transform(IMAGE_HEIGHT=HEIGHT, IMAGE_WIDTH=WIDTH))#mask_train_transform(IMAGE_HEIGHT=HEIGHT, IMAGE_WIDTH=WIDTH))

    dataloader = DataLoader(Plants, batch_size=4, shuffle=False)

    img_example, mask_example = Plants.__getitem__(7)


    plt.title('Representation of Sample Training Image and Instance Mask')

    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(np.array(img_example.permute(1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.title('Mask')
    plt.imshow(mask_example.permute(1,2,0))

    print(torch.unique(mask_example))
    plt.show()


if __name__ == '__main__':
    test()
