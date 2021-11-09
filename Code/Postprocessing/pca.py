import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import plotly.express as px
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple
from Preprocessing.dataset_plants_binary import CustomDatasetBinary
from Preprocessing.plant_transforms import image_train_transform, mask_train_transform

from Postprocessing.utils_post import create_masks_for_instance_N

loaded_model = torch.load('/Users/luisa/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/time_evolution/epoch-200.pt.nosync')
loaded_model.eval()

HEIGHT, WIDTH = 256, 256

directory = '/Users/luisa/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'

Plants = CustomDatasetMultiple(dir = directory,
                               transform = None,
                               image_transform= image_train_transform(HEIGHT, WIDTH),
                               mask_transform= mask_train_transform(HEIGHT, WIDTH))

img_example, mask_example = Plants.__getitem__(9)

image = img_example.unsqueeze(0)
mask = mask_example
embedding = loaded_model(image).squeeze(0)

#plt.imshow(image.detach().numpy().squeeze(0).T)
#plt.show()
print(embedding.shape)

flat_embedding = embedding.detach().numpy().reshape((16, -1)) #reshape?
print(flat_embedding.shape)
flat_mask = mask.reshape(-1)
C = len(np.unique(flat_mask))

values, counts = np.unique(flat_mask, return_counts=True)
values = np.array(values, dtype=int)

masks = np.array([create_masks_for_instance_N(mask, i) for i in values])
instances = np.array([flat_embedding*masks[i].reshape((16, -1)) for i in values])

#________________
input_for_PCA = flat_embedding.transpose()

scaler = StandardScaler()
input_for_PCA = scaler.fit_transform(input_for_PCA).transpose()

#scaled_instances = np.array([input_for_PCA*masks[i].reshape((16, -1)) for i in values])

print(input_for_PCA.shape)

dimension_a = 0
dimension_b = 1
dimension_c = 2

"""
fig = plt.figure(figsize = (20,10))
ax = fig.add_subplot(1, 2, 1, projection='3d')

plt.title('First three dimensions of embedding: no PCA')
for i in values[1:]:
    ax.scatter3D(instances[i][dimension_a], instances[i][dimension_b], instances[i][dimension_c], label = 'Instance {}'.format(i))

plt.title('First three dimensions of embedding (SCALED): no PCA')
for i in values[1:]:
    ax.scatter3D(scaled_instances[i][dimension_a], scaled_instances[i][dimension_b], scaled_instances[i][dimension_c], label = 'Instance {}'.format(i))
plt.show()
"""

pca = PCA(n_components = 3)
X = pca.fit_transform(input_for_PCA.transpose()).reshape(3, -1)
print('PCA done')
print('Shape:', X.shape)
print('Explained Variance', pca.explained_variance_ratio_)

df = pd.DataFrame()
df["y"] = flat_mask[:]
df["PCA_dim_1"] = X[0][:]
df["PCA_dim_2"] = X[1][:]
df["PCA_dim_3"] = X[2][:]

print(df.head(5))

#df = df.drop('y' == 0.0)

#plt.figure(figsize = (15, 15))
#sns.scatterplot(x="TSNE_dim_1", y="TSNE_dim_2", hue=df.y.tolist(),
                #palette=sns.color_palette("hls", C ),
                #data=df).set(title="data T-SNE projection")
#plt.show()
#plt.imshow(df.PCA_dim_1)

max_scaler = MinMaxScaler()

R_scaled = max_scaler.fit_transform(np.array(df.PCA_dim_1).reshape((-1, 1)))
G_scaled = max_scaler.fit_transform(np.array(df.PCA_dim_2).reshape((-1, 1)))
B_scaled = max_scaler.fit_transform(np.array(df.PCA_dim_3).reshape((-1, 1)))

R_channel = np.array(R_scaled).reshape((256,256))
G_channel = np.array(G_scaled).reshape((256,256))
B_channel = np.array(B_scaled).reshape((256,256))

PCA_RGB = np.dstack((R_channel,G_channel,B_channel))
#plt.imshow(PCA_RGB)
#plt.show()

#fig = px.scatter_3d(df, x='PCA_dim_1', y='PCA_dim_2', z='PCA_dim_3', color='y', symbol='y') #, size = 'y'
#fig.show()