import torch
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import os
from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple, image_train_transform, mask_train_transform
import random

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    src.to(DEVICE)
    other.to(DEVICE)
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    src.to(DEVICE)
    index.to(DEVICE)
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    src.to(DEVICE)
    index.to(DEVICE)

    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:

    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.floor_divide_(count)
    return out


def load_image_mask(height = 512, index = 0, mode = 'val'):
    HEIGHT, WIDTH = height, height

    rel_path = '~/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'
    directory = os.path.expanduser(rel_path)

    Plants = CustomDatasetMultiple(dir=directory,
                                   transform=None,
                                   image_transform=image_train_transform(HEIGHT, WIDTH),
                                   mask_transform=mask_train_transform(HEIGHT, WIDTH))

    torch.manual_seed(0)
    random.seed(0)
    train_set, val_set, test_set = torch.utils.data.random_split(Plants, [80, 28, 20])

    if mode == 'test':
        img_example, mask_example = test_set.__getitem__(index)
    if mode == 'train':
        img_example, mask_example = train_set.__getitem__(index)
    else:
        img_example, mask_example = val_set.__getitem__(index)

    image = img_example.unsqueeze(0)
    mask = mask_example  # want semantic mask instead of mask
    return image, mask

def save_embedding(embedding, output_path):

    raise NotImplementedError

    if len(embedding.size()) == 4:
        print(embedding.size())
        print('Batch')

    else:
        print('Single Embedding')
        embedding = embedding.squeeze(0).detach().numpy()

    E_DIM = embedding.detach().numpy().size
    print(E_DIM)

    flat_embedding = embedding.reshape((16, -1))

    print(flat_embedding.shape)

    np.savetxt(output_path, flat_embedding, delimiter=",")
    print('done')


def plot_results_from_training(epoch, losses, val_losses, title=None, save_path = None):
    plt.plot(np.linspace(1, epoch, epoch), losses / 80, label='Train Loss')
    plt.plot(np.linspace(1, epoch, epoch), val_losses/ 28, label='Validation Loss')

    if title is not None:
        plt.title('Train and Validation Loss Per Epoch per Image-Mask-Sample '+ str(title))
    else:
        plt.title('Train and Validation Loss Per Epoch per Image-Mask-Sample')

    plt.grid()
    plt.xlabel('Training Epoch')
    plt.ylabel('Training Loss')
    plt.yscale('log')
    plt.legend(borderpad=True)

    plt.savefig(save_path+'/loss_statistic.png', dpi=200)


if  __name__ == '__main__':
    random_embedding = torch.rand((1,400, 400, 16))

    save_embedding(random_embedding, output_path=None)