from utils import scatter_mean
import torch
import numpy as np


def compute_cluster_means(embeddings, target, n_instances):
    """
    Computes mean embeddings per instance.
    E - embedding dimension
    Args:
        embeddings: tensor of pixel embeddings, shape: ExSPATIAL
        target: one-hot encoded target instances, shape: SPATIAL
        n_instances: number of instances
    """
    target = target.type(torch.LongTensor)
    assert target.dtype == torch.int64, 'Target Mask does not have the right dtype; it should be torch.int64'

    embeddings = embeddings.flatten(1)  # 16-dim embedding [batch_size, 16, HEIGHT, WIDTH]
    target = target.flatten()

    mean_embeddings = scatter_mean(embeddings, target, dim_size = n_instances)
    return mean_embeddings.transpose(1, 0)


def test():
    # example:
    torch.manual_seed(3)

    random_prediction = torch.rand(16, 50, 50) * 255  # only 2 embedding dimensions [2, Height, Width]
    random_mask_tensor = torch.randint(low=0, high=2, size=(1, 50, 50), dtype = torch.int64)  # [1, Height, Width] # =target

    print(random_prediction.shape)
    print(random_mask_tensor.shape)

    instance_ids, instance_counts = torch.unique(random_mask_tensor,
                                                 return_counts=True)  # instance_ids = tensor([0,1]), instance_counts = 2

    C = instance_ids.size(0)  # number of instances, or alternatively instance_counts
    assert random_prediction.size()[1:] == random_mask_tensor.size()[1:]

    random_mask_tensor = random_mask_tensor.squeeze()  # --> [Height, Width]
    random_means = compute_cluster_means(random_prediction, random_mask_tensor, n_instances=C)
    return random_means


if __name__ == "__main__":
    test()
