import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple, mask_train_transform, image_train_transform
from Postprocessing.utils_post import create_masks_for_instance_N

loaded_model = torch.load('/Users/luisa/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/time_evolution/epoch-200.pt.nosync')
loaded_model.eval()

HEIGHT, WIDTH = 500, 500
PCA_dim = 3

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

embedding = embedding.detach().numpy() #.T #.reshape((-1, 16))
print(embedding.shape)
plt.imshow(embedding[1], cmap = 'hot')
plt.title('Information encoded in first embedding dimension')
plt.show()

"""Flatten Image"""
flat_embedding = embedding.reshape((16, -1))

"""1D to 2D: check if transformation works"""
embedding = flat_embedding.reshape((16, HEIGHT, WIDTH))
#plt.imshow(embedding[1], cmap = 'gray')
#plt.show()

"""Defining Scaler and Dimension Reduction Technique"""

scaler = StandardScaler()
pca = PCA(n_components = PCA_dim)

"""Fitting Scaler"""
input_for_PCA = scaler.fit_transform(flat_embedding)
print(input_for_PCA.shape)

"""Check if image still looks nice"""
check = input_for_PCA.reshape((16, HEIGHT, WIDTH))
#plt.imshow(check[1], cmap = 'gray')
#plt.show()

"""Do PCA"""
output_PCA = pca.fit_transform(input_for_PCA.T).T

image_PCA = output_PCA.reshape((PCA_dim, HEIGHT, WIDTH))
print(image_PCA.shape)

red_channel = image_PCA[0]
green_channel = image_PCA[1]
blue_channel = image_PCA[2]
print(red_channel.shape)

plt.imshow(red_channel, cmap = 'hot')
plt.title('Information in first PCA dimension')
plt.show()

"""Give PCA Statistic"""
print('explained variance per reduced dimension:', pca.explained_variance_ratio_)
print('cumulative explained variance:', np.cumsum(pca.explained_variance_ratio_))

"""MinMax Scaling all channels"""
max_scaler = MinMaxScaler()

red_channel = max_scaler.fit_transform(red_channel.reshape(-1,1)).reshape(HEIGHT, WIDTH)
green_channel = max_scaler.fit_transform(green_channel.reshape(-1,1)).reshape(HEIGHT,WIDTH)
blue_channel = max_scaler.fit_transform(blue_channel.reshape(-1,1)).reshape(HEIGHT,WIDTH)


"""Stack to RGB image"""
RGB_image = np.dstack((red_channel,green_channel,blue_channel))
plt.imshow(RGB_image)
plt.title('RGB image of first three PCA dimensions')
plt.show()
#red_channel = np.reshape(output_PCA.T[0], (HEIGHT, WIDTH))
#green_channel = output_PCA.T[1]
#blue_channel = output_PCA.T[2]

#print(red_channel.shape)

#plt.imshow(red_channel, cmap = 'gray' )
#plt.show()
