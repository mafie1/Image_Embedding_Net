import numpy as np
import torch
import matplotlib.pyplot as plt

from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple, mask_train_transform, image_train_transform
from Postprocessing.utils_post import create_masks_for_instance_N

loaded_model = torch.load('/Users/luisa/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/time_evolution/epoch-200.pt.nosync')
loaded_model.eval()

HEIGHT, WIDTH = 10, 10

directory = '/Users/luisa/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'

Plants = CustomDatasetMultiple(dir = directory,
                               transform = None,
                               image_transform= image_train_transform(HEIGHT, WIDTH),
                               mask_transform= mask_train_transform(HEIGHT, WIDTH))

img_example, mask_example = Plants.__getitem__(3)

image = img_example.unsqueeze(0)
mask = mask_example
embedding = loaded_model(image).squeeze(0)
print(embedding.shape)

embedding = embedding.detach().numpy().flatten() #.T #.reshape((-1, 16))

np.savetxt("sample_embedding{}.csv".format(HEIGHT), embedding, delimiter=",")
