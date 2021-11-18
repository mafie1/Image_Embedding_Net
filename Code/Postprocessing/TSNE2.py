import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple, mask_train_transform, image_train_transform
from Postprocessing.utils_post import create_masks_for_instance_N


loaded_model = torch.load('/Users/luisa/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/time_evolution/epoch-1000.pt.nosync')
loaded_model.eval()

HEIGHT, WIDTH = 400, 400
DIM_RED = 3

directory = '/Users/luisa/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'

Plants = CustomDatasetMultiple(dir = directory,
                               transform = None,
                               image_transform= image_train_transform(HEIGHT, WIDTH),
                               mask_transform= mask_train_transform(HEIGHT, WIDTH))

img_example, mask_example = Plants.__getitem__(20)

image = img_example.unsqueeze(0)
mask = mask_example
embedding = loaded_model(image).squeeze(0)

embedding = embedding.detach().numpy() #.T #.reshape((-1, 16))

plt.imshow(embedding[1], cmap = 'hot')
#plt.title('Information encoded in first embedding dimension')
plt.xticks([])
plt.yticks([])
plt.savefig('First_embedding_dim.png', dpi = 200)
plt.show()

"""Flatten Image"""
flat_embedding = embedding.reshape((16, -1))

"""1D to 2D: check if transformation works"""
embedding = flat_embedding.reshape((16, HEIGHT, WIDTH))

"""Defining scaler and dimension reduction technique"""
scaler = StandardScaler()
tsne = TSNE(n_components = DIM_RED)

"""Preparing input for Dimension Reduction technique"""
input = scaler.fit_transform(flat_embedding)
print(input.shape)

"""Check if image still looks nice"""
check = input.reshape((16, HEIGHT, WIDTH))
plt.imshow(check[1], cmap = 'hot')
plt.xticks([])
plt.yticks([])
#plt.title('Information in first embedding dimension after rescaling')
plt.savefig('First_TSNE.png', dpi =200)
plt.show()

"""DO TSNE"""
output = tsne.fit_transform(input.T).T

image_TSNE = output.reshape((DIM_RED, HEIGHT, WIDTH))
print(image_TSNE.shape)

red_channel = image_TSNE[0]
green_channel = image_TSNE[1]
blue_channel = image_TSNE[2]

plt.imshow(red_channel, cmap = 'hot')
plt.title('Information in first TSNE dimension')
plt.show()
