import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from plant_transforms import mask_train_transform, image_train_transform


class CustomDataset(Dataset):
    def __init__(self, dir, transform=None, image_transform=None, mask_transform=None):
        self.directory = dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_transform = image_transform
        self.contents = os.listdir(dir)
        self.images = list(filter(lambda k: 'rgb' in k, self.contents))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.directory, self.images[index])
        mask_path = os.path.join(self.directory, self.images[index].replace('rgb.png', 'fg.png'))
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = np.array(mask / 255, dtype=np.int64)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask


if __name__ == '__main__':
    SIZE = 256
    image_directory = '/Users/luisa/Documents/BA_Thesis/CVPPP2015_LCC_training_data/A1'
    Plants = CustomDataset(image_directory,
                           image_transform=image_train_transform(SIZE, SIZE),
                           mask_transform=mask_train_transform(SIZE, SIZE))

    dataloader = DataLoader(Plants,
                            batch_size=4,
                            shuffle=False)

    img_example, mask_example = Plants.__getitem__(3)

    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(img_example)

    plt.subplot(1, 2, 2)
    plt.title('Mask')
    plt.imshow(mask_example)
    plt.show()

# print(f'Train dataset has {len(dataloader)} batches of size {4}')
