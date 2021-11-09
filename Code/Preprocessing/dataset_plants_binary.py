import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Code.Preprocessing.plant_transforms import image_train_transform, mask_train_transform


class CustomDatasetBinary(Dataset):
    def __init__(self, dir, transform, image_transform, mask_transform):
        self.directory = dir
        self.transform = transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.contents = os.listdir(dir)
        self.images = list(filter(lambda k: 'rgb' in k, self.contents))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.directory, self.images[index])
        mask_path = os.path.join(self.directory, self.images[index].replace('rgb.png', 'fg.png'))

        #image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype = np.float32)

        uniques = np.unique(mask)
        integers = np.linspace(0, len(uniques) - 1, len(uniques))

        for i in range(0, len(uniques)):
            mask = np.where(mask == uniques[i], integers[i], mask)

        #mask = np.array(mask, dtype=np.int64)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask


def test():
    img_path = '/Users/luisa/Documents/BA_Thesis/Datasets for Multiple Instance Seg/CVPPP2017_instances/training/A1'
    # mask_path = '/Users/luisa/Documents/BA_Thesis/Datasets for Multiple Instance Seg/CVPPP2017_instances/training/A1/plant001_label.png'

    HEIGHT = 150
    WIDTH = 150

    Plants = CustomDatasetBinary(dir=img_path,
                                   transform=None,
                                   image_transform=image_train_transform(IMAGE_HEIGHT = HEIGHT, IMAGE_WIDTH=WIDTH),
                                   mask_transform=mask_train_transform(IMAGE_HEIGHT=HEIGHT, IMAGE_WIDTH=WIDTH))#mask_train_transform(IMAGE_HEIGHT=HEIGHT, IMAGE_WIDTH=WIDTH))

    dataloader = DataLoader(Plants, batch_size=4, shuffle=False)

    img_example, mask_example = Plants.__getitem__(7)

    print(mask_example.shape)
    print(img_example.shape)

    plt.title('Representation of Sample Training Image and Binary Segmentation Mask')

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
