import numpy as np
import torch
import random
import os
from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple, image_train_transform, mask_train_transform

def create_masks_for_instance_N(mask, N):
    flat_mask = mask.reshape(-1)
    C = len(np.unique(flat_mask))
    assert N <= C

    mask_4 = torch.stack((flat_mask, flat_mask, flat_mask, flat_mask))
    mask_16 = torch.vstack((mask_4, mask_4, mask_4, mask_4)).reshape((16, -1)).detach().numpy()

    example_mask_N = np.where(mask_16 == np.unique(mask_16)[N], 1, 0).reshape((16, -1))
    return example_mask_N


def load_val_image(height, index):
    HEIGHT = height
    WIDTH = height

    rel_path = '~/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'
    directory = os.path.expanduser(rel_path)

    Plants = CustomDatasetMultiple(dir=directory,
                                   transform=None,
                                   image_transform=image_train_transform(HEIGHT, WIDTH),
                                   mask_transform=mask_train_transform(HEIGHT, WIDTH))

    torch.manual_seed(0)
    random.seed(0)
    train_set, val_set, test_set = torch.utils.data.random_split(Plants, [80, 28, 20])

    img_example, mask_example = val_set.__getitem__(index)

    image = img_example.unsqueeze(0)
    mask = mask_example  # want semantic mask instead of mask

    return image, mask