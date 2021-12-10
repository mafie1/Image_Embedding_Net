import torch
import numpy as np
import matplotlib.pyplot as plt

from Custom_Loss.cluster_means import compute_cluster_means

# here begins the actual variances part
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_cluster_variances(cluster_means, embedding, target, instance_counts, delta_var = 0.5, ignore_zero_label = False):
    """
    :param cluster_means: output of the function compute_cluster_means
    :param embedding: prediction from neural network [16, H, W]
    :param target: instance segmentation mask, one-hot encoded [1, H, W]
    :param instance_counts: number of pixels per instance = tensor([num in instance 1, num in instance 2, ...])
    :param ignore_zero: ignore the background when calculating the loss, default = True
    :return: tensor(vector) with variance per instance
    """
    cluster_means.to(DEVICE)
    embedding.to(DEVICE)
    target.to(DEVICE)
    instance_counts.to(DEVICE)
    assert target.dim() == 2
    C = cluster_means.shape[0]  # number of instances C = 2 in example

    cluster_means_spatial = cluster_means[target]  # projects the cluster means on the mask
    cluster_means_spatial = cluster_means_spatial.permute(2, 0, 1)

    instance_sizes_spatial = instance_counts[target]  # project the number of pixels in each instance on the mask

    #dist_to_mean = torch.norm(embedding - cluster_means_spatial, p='fro', dim=0)
    dist_to_mean = torch.linalg.norm(embedding - cluster_means_spatial, dim = 0)


    hinge_dist = torch.clamp(dist_to_mean - delta_var,
                             min=0)  # eliminate all distances less than delta_var (=within the cluster)
    hinge_dist = hinge_dist ** 2

    variance = torch.sum(hinge_dist.to(DEVICE) / instance_sizes_spatial.to(DEVICE)) / C

    return variance


if __name__ == "__main__":
    variances = compute_cluster_variances(random_means, random_prediction, random_mask_tensor, instance_counts)

    mean1, mean2 = random_means[0], random_means[1]
    variance2 = float(variances)

    instance_1 = np.where(random_mask_tensor == 1, random_prediction, 0)
    instance_2 = np.where(random_mask_tensor == 0, random_prediction, 0)

    x = random_prediction[0]
    y = random_prediction[1]

    print(random_mask_tensor)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Mask and Scatter Plot of Random 2d embedding')

    #ax1.scatter(x, y, label='Random Embedding of Pixels')
    ax1.scatter(instance_1[0], instance_1[1], label = 'instance 1')
    ax1.scatter(instance_2[0], instance_2[1], label='instance 2', c = 'orange')
    ax1.scatter(mean2[0], mean2[1], label = 'mean of instance 2', c = 'red')

    circle1 = plt.Circle((mean2[0], mean2[1]), variance2**0.5, color='r', fill = False, label = 'variance of mean 2')

    ax1.add_patch(circle1)
    ax1.set_ylim(0, 255)
    ax1.set_xlim(0, 255)
    ax1.legend()
    ax2.imshow(random_mask_tensor)
    plt.show()
