import torch
import numpy as np
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.metrics import adapted_rand_error, contingency_table


def precision(tp, fp, fn):
    return tp / (tp + fp) if tp > 0 else 0


def recall(tp, fp, fn):
    return tp / (tp + fn) if tp > 0 else 0


def accuracy(tp, fp, fn):
    return tp / (tp + fp + fn) if tp > 0 else 0


def f1(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0


def _relabel(input):
    _, unique_labels = np.unique(input, return_inverse=True)
    return unique_labels.reshape(input.shape)


def _iou_matrix(gt, seg):
    # relabel gt and seg for smaller memory footprint of contingency table
    gt = _relabel(gt)
    seg = _relabel(seg)

    # get number of overlapping pixels between GT and SEG
    n_inter = contingency_table(gt, seg).A

    # number of pixels for GT instances
    n_gt = n_inter.sum(axis=1, keepdims=True)
    # number of pixels for SEG instances
    n_seg = n_inter.sum(axis=0, keepdims=True)

    # number of pixels in the union between GT and SEG instances
    n_union = n_gt + n_seg - n_inter

    iou_matrix = n_inter / n_union
    # make sure that the values are within [0,1] range
    assert 0 <= np.min(iou_matrix) <= np.max(iou_matrix) <= 1

    return iou_matrix


class SegmentationMetrics:
    """
    Computes precision, recall, accuracy, f1 score for a given ground truth and predicted segmentation.
    Contingency table for a given ground truth and predicted segmentation is computed eagerly upon construction
    of the instance of `SegmentationMetrics`.
    Args:
        gt (ndarray): ground truth segmentation
        seg (ndarray): predicted segmentation
    """

    def __init__(self, gt, seg):
        self.iou_matrix = _iou_matrix(gt, seg)

    def metrics(self, iou_threshold):
        """
        Computes precision, recall, accuracy, f1 score at a given IoU threshold
        """
        # ignore background
        iou_matrix = self.iou_matrix[1:, 1:]
        detection_matrix = (iou_matrix > iou_threshold).astype(np.uint8)
        n_gt, n_seg = detection_matrix.shape

        # if the iou_matrix is empty or all values are 0
        trivial = min(n_gt, n_seg) == 0 or np.all(detection_matrix == 0)
        if trivial:
            tp = fp = fn = 0
        else:
            # count non-zero rows to get the number of TP
            tp = np.count_nonzero(detection_matrix.sum(axis=1))
            # count zero rows to get the number of FN
            fn = n_gt - tp
            # count zero columns to get the number of FP
            fp = n_seg - np.count_nonzero(detection_matrix.sum(axis=0))

        return {
            'precision': precision(tp, fp, fn),
            'recall': recall(tp, fp, fn),
            'accuracy': accuracy(tp, fp, fn),
            'f1': f1(tp, fp, fn)
        }

class AveragePrecision:
    """
    Average precision taken for the IoU range (0.5, 0.95) with a step of 0.05 as defined in:
    https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric
    """

    def __init__(self):
        self.iou_range = np.linspace(0.50, 0.95, 10)

    def __call__(self, input_seg, gt_seg):
        # compute contingency_table
        sm = SegmentationMetrics(gt_seg, input_seg)
        # compute accuracy for each threshold
        acc = [sm.metrics(iou)['accuracy'] for iou in self.iou_range]
        # return the average
        return np.mean(acc)

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

    return np.abs(N_pred - N_GT)


def DICE_score(pred, ground_truth):
    assert pred.shape == ground_truth.shape

    #pred, ground_truth = multi_class_to_one_hot(pred), multi_class_to_one_hot(ground_truth)

    nom = 2 * torch.sum(pred * ground_truth)

    denom = torch.sum(ground_truth) + torch.sum(pred)

    dice = float(nom) / float(denom)

    return dice


def calc_DiC(pred_seg, ground_truth):
    n_objects_gt = np.unique(ground_truth)
    n_objects_pred = np.unique(pred_seg)

    return np.abs(n_objects_gt - n_objects_pred)


def calc_dice(gt_seg, pred_seg):

    nom = 2 * np.sum(gt_seg * pred_seg)
    denom = np.sum(gt_seg) + np.sum(pred_seg)

    dice = float(nom) / float(denom)
    return dice


def calc_bd(ins_seg_gt, ins_seg_pred):

    gt_object_idxes = list(set(np.unique(ins_seg_gt)).difference([0]))
    pred_object_idxes = list(set(np.unique(ins_seg_pred)).difference([0]))

    best_dices = []
    for gt_idx in gt_object_idxes:
        _gt_seg = (ins_seg_gt == gt_idx).astype('bool')
        dices = []
        for pred_idx in pred_object_idxes:
            _pred_seg = (ins_seg_pred == pred_idx).astype('bool')

            dice = calc_dice(_gt_seg, _pred_seg)
            dices.append(dice)
        best_dice = np.max(dices)
        best_dices.append(best_dice)

    best_dice = np.mean(best_dices)

    return best_dice


def get_SBD(ins_seg_gt, ins_seg_pred):

    _dice1 = calc_bd(ins_seg_gt, ins_seg_pred)
    _dice2 = calc_bd(ins_seg_pred, ins_seg_gt)
    return min(_dice1, _dice2)

"""
if opt.dataset == 'CVPPP':
    names = np.loadtxt('../data/metadata/CVPPP/validation_image_paths.txt',
                       dtype='str', delimiter=',')
    names = np.array([os.path.splitext(os.path.basename(n))[0] for n in names])
    n_objects_gts = np.loadtxt(
        '../data/metadata/CVPPP/number_of_instances.txt',
        dtype='str',
        delimiter=',')
    img_dir = '../data/raw/CVPPP/CVPPP2017_LSC_training/training/A1'

    dics, sbds, fg_dices = [], [], []
    for name in names:
        if not os.path.isfile(
                '{}/{}/{}-n_objects.npy'.format(pred_dir, name, name)):
            continue

        n_objects_gt = int(n_objects_gts[n_objects_gts[:, 0] == name.replace('_rgb', '')][0][1])
        n_objects_pred = np.load(
            '{}/{}/{}-n_objects.npy'.format(pred_dir, name, name))

        ins_seg_gt = np.array(Image.open(
            os.path.join(img_dir, name.replace('_rgb', '') + '_label.png')))
        ins_seg_pred = np.array(Image.open(os.path.join(
            pred_dir, name, name + '-ins_mask.png')))

        fg_seg_gt = np.array(
            Image.open(
                os.path.join(
                    img_dir,
                    name.replace('_rgb', '') +
                    '_fg.png')))
        fg_seg_pred = np.array(Image.open(os.path.join(
            pred_dir, name, name + '-fg_mask.png')))

        fg_seg_gt = (fg_seg_gt == 1).astype('bool')
        fg_seg_pred = (fg_seg_pred == 255).astype('bool')

        sbd = calc_sbd(ins_seg_gt, ins_seg_pred)
        sbds.append(sbd)

        dic = calc_dic(n_objects_gt, n_objects_pred)
        dics.append(dic)

        fg_dice = calc_dice(fg_seg_gt, fg_seg_pred)
        fg_dices.append(fg_dice)

    mean_dic = np.mean(dics)
    mean_sbd = np.mean(sbds)
    mean_fg_dice = np.mean(fg_dices)

    print 'MEAN SBD     : ', mean_sbd
    print 'MEAN |DIC|   : ', mean_dic
    print 'MEAN FG DICE : ', mean_fg_dice
    """



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
