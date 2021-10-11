import numpy as np
import torch


def create_masks_for_instance_N(mask, N):
    flat_mask = mask.reshape(-1)
    C = len(np.unique(flat_mask))
    assert N <= C

    mask_4 = torch.stack((flat_mask, flat_mask, flat_mask, flat_mask))
    mask_16 = torch.vstack((mask_4, mask_4, mask_4, mask_4)).reshape((16, -1)).detach().numpy()

    example_mask_N = np.where(mask_16 == np.unique(mask_16)[N], 1, 0).reshape((16, -1))
    return example_mask_N