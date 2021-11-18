import numpy as np
import torch
import torch.nn as nn

class AffinityLoss(nn.Module):
    def __init__(self, lamb):
        super(AffinityLoss, self).__init__()
        self.lamb = lamb

    def forward(self, batch_embedding, batch_target):
        initial_loss = 0
        n_batches = batch_embedding.shape[0]

        for single_input, single_target in zip(batch_embedding,
                                               batch_target):

        return 0


class AffinityLayer(nn.Module):
    def __init__(self, sigma, offset):
        super(AffinityLayer,self).__init__()
        self.sigma = sigma
        self.offset = offset

    def get_affinities(self, emb_1, emb_2, aff_measure):
        #affinity = aff_measure(emb_1, emb_2)
        affinity = 2/(1+np.exp(self.sigma * np.abs(emb_1 - emb_2)**2))
        pass

    def average_aff(self):
        pass

    def forward(self):
        get_affinities()
        average_aff()
        pass



