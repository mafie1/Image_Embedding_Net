import torch
import torch.optim as optim

IN_CHANNELS = 3
OUT_CHANNELS = 16

DELTA_VAR = 0.5
DELTA_D = 1.5

EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.001

HEIGHT, WIDTH = 128, 128


#OPTIMIZER = optim.Adam()
#OPTIMIZER = optim.SGD()