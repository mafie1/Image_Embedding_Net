import torch
import torch.nn as nn

from Custom_Loss.cluster_means import compute_cluster_means
from Custom_Loss.cluster_variance import compute_cluster_variances
from Custom_Loss.regularization_term import compute_regularizer_term
from Custom_Loss.distance_term import compute_distance_term

from model import UNet_spoco

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def push_pull_loss(batch_embedding, batch_target, alpha=1, beta=1, gamma=0.001, delta_var = 0.5, delta_d = 2.5):
    """
    The push_pull loss applies to an entire training batch.
    Inputs:
        batch_embedding: shape is [batchsize, ]
    """
    n_batches = batch_embedding.shape[0]

    loss = 0

    for single_input, single_target in zip(batch_embedding,
                                           batch_target):  # calculate loss for every embedding-mask pair in the batch
        contains_bg = 0 in single_target
        single_target = single_target.squeeze()
        single_target = single_target.type(torch.LongTensor)

        assert single_target.dtype == torch.int64, 'Target Mask does not have the right dtype; it should be torch.int64'

        #single_target = single_target.int(torch.)
        # if self.unlabeled_push and contains_bg:
        #   ignore_zero_label = True

        instance_ids, instance_counts = torch.unique(single_target, return_counts=True)
        C = instance_ids.size(0)

        # assert single_input.size()[1:] == single_target.squeeze().size()  #ATTENTION: what format should the target ideally be in?

        cluster_means = compute_cluster_means(single_input, single_target, C)

        variance_term = compute_cluster_variances(cluster_means, single_input, single_target, instance_counts, delta_var=delta_var)
        regularization_term = compute_regularizer_term(single_input, single_target)
        distance_term = compute_distance_term(single_input, single_target, delta_d=delta_d)

        #print('')
        #print('Variance Term: ', alpha * variance_term)
        #print('Distance_term: ', beta * distance_term)
        #print('Regularization Term:', gamma*regularization_term)

        loss = alpha * variance_term + beta * distance_term + gamma * regularization_term

        return loss


class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_var = 0.5, delta_d = 2.5, alpha = 1.0, beta = 1.0, gamma = 0.001):

        super(DiscriminativeLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_var = delta_var
        self.delta_d = delta_d

    def forward(self, batch_embedding, batch_target):
        loss = push_pull_loss(batch_embedding, batch_target, self.alpha, self.beta, self.gamma, delta_var=self.delta_var, delta_d=self.delta_d)
        return loss


def test():
    # this time, create a batch with batch_size = 2
    HEIGHT = 150
    WIDTH = 150
    E_DIM = 16

    model = UNet_spoco(in_channels=3, out_channels=2)

    random_image = torch.rand(1, 3, HEIGHT, WIDTH) * 255
    prediction_to_image = model(random_image)

    random_prediction_1 = torch.rand(16, HEIGHT, WIDTH) * 255  # 16 embedding dimensions [16, Height, Width]
    random_mask_tensor_1 = torch.randint(low=0, high=2, size=(1, HEIGHT, WIDTH))

    random_prediction_2 = torch.rand(16, HEIGHT, WIDTH) * 255  # 16 embedding dimensions [16, Height, Width]
    random_mask_tensor_2 = torch.randint(low=0, high=2, size=(1, HEIGHT, WIDTH))

    batch_prediction = torch.stack((random_prediction_1, random_prediction_2))
    batch_mask = torch.stack((random_mask_tensor_1, random_mask_tensor_2))

    loss_function = DiscriminativeLoss(delta_var=2.0, delta_d=5.0)

    print(batch_prediction.shape)
    print(batch_mask.shape)

    print(loss_function(batch_prediction, batch_mask))


def test2():
    from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple, image_train_transform, mask_train_transform
    import matplotlib.pyplot as plt

    directory = '/Users/luisa/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'

    HEIGHT, WIDTH = 50, 50

    Plants = CustomDatasetMultiple(dir=directory,
                                   transform=None,
                                   image_transform=image_train_transform(HEIGHT, WIDTH),
                                   mask_transform=mask_train_transform(HEIGHT, WIDTH))

    img_example, mask_example = Plants.__getitem__(7)
    img_example = img_example.unsqueeze(0)
    mask_example = mask_example.unsqueeze(0)

    model = UNet_spoco(in_channels=3, out_channels=16)
    prediction = model(img_example)


    loss_function = DiscriminativeLoss(delta_var=0.5)

    loss = loss_function(prediction, mask_example)

    prediction = prediction.detach().numpy()
    plt.scatter(prediction[0][0], prediction[0][1])
    plt.show()

    print(loss)



if __name__ == '__main__':
    test2()
