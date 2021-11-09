from Custom_Loss.cluster_means import compute_cluster_means
import torch


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

    #if C <= 1:
     #   return 0.

    cluster_means = cluster_means.unsqueeze(0)
    shape = list(cluster_means.size())

    cm_matrix1 = cluster_means.expand(shape)
    cm_matrix2 = cm_matrix1.permute(1, 0, 2)

    dist_matrix = torch.linalg.norm(cm_matrix1 - cm_matrix2, dim=2)
    repulsion_dist = 2 * delta_d * (1 - torch.eye(C))
    #torch.eye creates unit diagronal matrix
    #repulsion_dist = repulsion_dist.to(DEVICE)

    if ignore_zero_label:
        if C == 2:
            return 0

        d_min = torch.min(dist_matrix[dist_matrix > 0]).item()

        # dist_multiplier = 2 * delta_dist / d_min + epsilon
        dist_multiplier = 2 * delta_d / d_min + 1e-3

        # create distance mask: this part I do not yet understand: is this some kind of normalization?
        dist_mask = torch.ones_like(dist_matrix)
        dist_mask[0, 1:] = dist_multiplier
        dist_mask[1:, 0] = dist_multiplier

        # mask the dist_matrix
        dist_matrix = dist_matrix * dist_mask
        # decrease number of instances
        C -= 1


    # zero out distances greater than 2*delta_dist (hinge)
    #hinged_dist = torch.clamp(dist_matrix - repulsion_dist, min=0) ** 2
    hinged_dist = torch.clamp(repulsion_dist- dist_matrix, min = 0)**2
    # sum all of the hinged pair-wise distances
    dist_sum = torch.sum(hinged_dist)
    distance_term = dist_sum / (C * (C - 1))
    return distance_term


def test():
    HEIGHT = 5
    WIDTH = 5
    torch.manual_seed(3)
    random_prediction = torch.rand(2, HEIGHT, WIDTH) * 255  # only 2 embedding dimensions [2, Height, Width]
    random_mask_tensor = torch.randint(low=0, high=2, size=(1, HEIGHT, WIDTH))  # [1, Height, Width]

    compute_distance_term(random_prediction, random_mask_tensor, delta_d= 2.)

def test2():
    #now test if it works with 2+ instances
    HEIGHT = 50
    WIDTH = 50
    torch.manual_seed(3)
    random_prediction = torch.rand(2,HEIGHT,WIDTH)*255
    random_mask_tensor = torch.randint(low= 0, high = 3, size = (1,HEIGHT,WIDTH))

    compute_distance_term(random_prediction, random_mask_tensor, delta_d = 2.)


if __name__ == '__main__':
    test2()



