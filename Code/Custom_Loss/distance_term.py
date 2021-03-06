from Custom_Loss.cluster_means import compute_cluster_means
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_distance_term(embedding, target, delta_d = 2.5, ignore_zero_label = False):
    """
    :param embedding:
    :param target:
    :param ignore_zero_label: default = True
    :param delta_d: distance parameter, 1.5-2.0
    :return
    """
    instance_idx, instance_counts = torch.unique(target, return_counts=True)
    C = instance_idx.size(0)
    cluster_means = compute_cluster_means(embedding, target.squeeze(), n_instances=C)

    cluster_means = cluster_means.unsqueeze(0)
    shape = list(cluster_means.size())

    cm_matrix1 = cluster_means.expand(shape)
    cm_matrix2 = cm_matrix1.permute(1, 0, 2)

    dist_matrix = torch.linalg.norm(cm_matrix1 - cm_matrix2, dim=2)
    repulsion_dist = 2 * delta_d * (1 - torch.eye(C))

    #torch.eye creates unit diagronal matrix
    #repulsion_dist = repulsion_dist.to(DEVICE)


    # zero out distances greater than 2*delta_dist (hinge)
    hinged_dist = torch.clamp(repulsion_dist.to(DEVICE) - dist_matrix.to(DEVICE), min = 0)**2

    # sum all of the hinged pair-wise distances
    dist_sum = torch.sum(hinged_dist)
    distance_term = dist_sum / (C * (C - 1))
    return distance_term



def test2():

    HEIGHT = 50
    WIDTH = 50
    torch.manual_seed(3)
    random_prediction = torch.rand(2, HEIGHT, WIDTH)*255
    random_mask_tensor = torch.randint(low= 0, high = 3, size = (1,HEIGHT,WIDTH))

    compute_distance_term(random_prediction, random_mask_tensor, delta_d=2.)


if __name__ == '__main__':
    test2()



