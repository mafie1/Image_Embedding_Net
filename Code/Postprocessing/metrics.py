import torch
import numpy as np
import random
import torch.nn as nn
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    Returns:
        Cx1 tensor, i.e. Dice score for each channel
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = torch.flatten(input)
    target = torch.flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


def counting_score(pred, ground_truth):

    assert pred.shape == ground_truth.shape

    N_pred = len(np.unique(pred))
    N_GT = len(np.unique(ground_truth))

    return np.abs(N_pred - N_GT)/N_GT

def ARI_score(pred, ground_truth):

    raise NotImplementedError
    #assert pred.shape == ground_truth.shape

    #ARI = adjusted_rand_score(pred, ground_truth)
    #return ARI

def multi_class_to_one_hot(tensor):

    one_hot_vector = torch.nn.functional.one_hot(tensor)
    return one_hot_vector


def DICE_score(pred, ground_truth):
    assert pred.shape == ground_truth.shape

    #pred, ground_truth = multi_class_to_one_hot(pred), multi_class_to_one_hot(ground_truth)

    nom = 2 * torch.sum(pred * ground_truth)

    denom = torch.sum(ground_truth) + torch.sum(pred)

    dice = float(nom) / float(denom)

    return dice

if __name__ == '__main__':
    torch.manual_seed(5)
    random.seed(5)

    max = np.random.randint(1, 4)
    print(max)

    prediction = torch.randint(low=0, high=3, size=(1, 4, 4,))
    ground_truth_image = torch.randint(low=0, high=max, size=(1, 4, 4))

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(prediction.squeeze())
    ax1.set_title('Prediction')
    ax2.imshow(ground_truth_image.squeeze())
    ax2.set_title('Ground Truth')

    plt.show()

    print(counting_score(prediction, ground_truth_image))

    #print('ARI:', ARI_score(prediction, ground_truth_image))

    print('Dice:', compute_per_channel_dice(prediction, ground_truth_image))
