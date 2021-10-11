from Custom_Loss.cluster_means import compute_cluster_means
import torch


def compute_regularizer_term(embedding, target):
    """
    Computes the regularizer term, i.e. a small pull-force that draws all clusters towards origin to keep
    the network activations bounded
    """

    instance_ids, instance_counts = torch.unique(target, return_counts=True)
    C = instance_ids.size(0)
    cluster_means = compute_cluster_means(embedding, target.squeeze(), n_instances=C)

    # compute the norm of the mean embeddings
    #norms = torch.norm(cluster_means, dim=1)
    norms = torch.linalg.norm(cluster_means, dim=0)
    # return the average norm per batch
    return torch.sum(norms) / cluster_means.size(0)


def test():
    HEIGHT = 50
    WIDTH = 50
    torch.manual_seed(3)
    random_prediction = torch.rand(2, HEIGHT, WIDTH) * 255
    random_mask_tensor = torch.randint(low=0, high=4, size=(1, HEIGHT, WIDTH))

    return compute_regularizer_term(random_prediction, random_mask_tensor)


if __name__ == '__main__':
    print(test())

