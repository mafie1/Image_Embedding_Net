from Custom_Loss.cluster_means import compute_cluster_means
import torch


def compute_regularizer_term(embedding, target):
    """
    This function computes the regularization term.
    This term prevents the clusters to drift off. With the regularization term,
    clusters are forced to stay close to the origin.
    """

    instance_idx, _ = torch.unique(target, return_counts=True)
    C = instance_idx.size(0)

    # computes the mean embedding of each instance cluster
    cluster_means = compute_cluster_means(embedding, target.squeeze(), n_instances=C)

    # computes the norm of the mean embeddings
    norms = torch.linalg.norm(cluster_means, dim=1)

    # computes the average norm, that is the norm for each instance divided by the total number of instances
    return torch.sum(norms)/ cluster_means.size(0)


def test():
    HEIGHT = 50
    WIDTH = 50
    torch.manual_seed(3)
    random_prediction = torch.rand(2, HEIGHT, WIDTH) * 255
    random_mask_tensor = torch.randint(low=0, high=5, size=(1, HEIGHT, WIDTH))

    print(compute_regularizer_term(random_prediction, random_mask_tensor))


if __name__ == '__main__':
    test()

