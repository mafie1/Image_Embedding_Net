import torch

torch.backends.cudnn.benchmark = True #may slow training at the beginning but will find best algorithm for convolutions

big = 'yes'
#big = None

LEARNING_RATE = 0.001  # 1e-3 empfohlen
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCHS =  3000
HEIGHT = 512
WIDTH = HEIGHT
IN_CHANNELS = 3
OUT_CHANNELS = 3  # output dimensions of embedding space

DELTA_VAR = 1.5
DELTA_D = 4
