import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple
from Preprocessing.dataset_plants_binary import CustomDatasetBinary
from Preprocessing.plant_transforms import image_train_transform, mask_train_transform

from Postprocessing.utils_post import create_masks_for_instance_N


loaded_model = torch.load('/Users/luisa/Documents/BA_Thesis/Embedding UNet/Code/saved_models/time_evolution/epoch-29.pt')
loaded_model.eval()

HEIGHT, WIDTH = 128, 128

directory = '/Users/luisa/Documents/BA_Thesis/Datasets for Multiple Instance Seg/CVPPP2017_instances/training/A1/'

Plants = CustomDatasetMultiple(dir = directory,
                               transform = None,
                               image_transform= image_train_transform(HEIGHT, WIDTH),
                               mask_transform= mask_train_transform(HEIGHT, WIDTH))

img_example, mask_example = Plants.__getitem__(5)

image = img_example.unsqueeze(0)
mask = mask_example

embedding = loaded_model(image).squeeze(0)
flat_embedding = embedding.detach().numpy().reshape((16, -1))
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
scaled_instances = np.array([input_for_PCA*masks[i].reshape((16, -1)) for i in values])



dimension_a = 0
dimension_b = 1
dimension_c = 2

#fig = plt.figure(figsize = (20,10))
#ax = fig.add_subplot(1, 2, 1, projection='3d')

"""
plt.title('First three dimensions of embedding: no PCA')
for i in values[1:]:
    ax.scatter3D(instances[i][dimension_a], instances[i][dimension_b], instances[i][dimension_c], label = 'Instance {}'.format(i))

plt.title('First three dimensions of embedding (SCALED): no PCA')
for i in values[1:]:
    ax.scatter3D(scaled_instances[i][dimension_a], scaled_instances[i][dimension_b], scaled_instances[i][dimension_c], label = 'Instance {}'.format(i))
plt.show()
"""

tsne = TSNE(n_components = 3, n_iter=250)
X = tsne.fit_transform(input_for_PCA.transpose()).reshape(3, -1)
print('TSNE done')
print('Shape:', X.shape)

df = pd.DataFrame()
df["y"] = flat_mask[:]
df["TSNE_dim_1"] = X[0][:]
df["TSNE_dim_2"] = X[1][:]
df["TSNE_dim_3"] = X[2][:]

print(df.head(5))

#df = df.drop('y' == 0.0)

#plt.figure(figsize = (15, 15))
#sns.scatterplot(x="TSNE_dim_1", y="TSNE_dim_2", hue=df.y.tolist(),
                #palette=sns.color_palette("hls", C ),
                #data=df).set(title="data T-SNE projection")
#plt.show()

fig = px.scatter_3d(df, x='TSNE_dim_1', y='TSNE_dim_2', z='TSNE_dim_3', color='y', symbol='y', size = 'y')
fig.show()