import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import dis

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple, mask_train_transform, image_train_transform
from Postprocessing.utils_post import create_masks_for_instance_N

loaded_model = torch.load('/Users/luisa/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/time_evolution/epoch-1000.pt.nosync')
loaded_model.eval()

HEIGHT, WIDTH = 400, 400
PCA_dim = 16

directory = '/Users/luisa/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'

Plants = CustomDatasetMultiple(dir = directory,
                               transform = None,
                               image_transform= image_train_transform(HEIGHT, WIDTH),
                               mask_transform= mask_train_transform(HEIGHT, WIDTH))

img_example, mask_example = Plants.__getitem__(20)

image = img_example.unsqueeze(0)
mask = mask_example
embedding = loaded_model(image).squeeze(0)
print(embedding.shape)

embedding = embedding.detach().numpy() #.T #.reshape((-1, 16))
print(embedding.shape)
plt.imshow(embedding[1], cmap = 'hot')
plt.xticks([])
plt.yticks([])
#plt.title('Information encoded in first embedding dimension')
#plt.savefig('First_dim.png', dpi = 200)
#plt.show()

"""Flatten Image"""
flat_embedding = embedding.reshape((16, -1))
flat_mask = mask.reshape(-1)


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

red_channel = image_PCA[0]
green_channel = image_PCA[1]
blue_channel = image_PCA[2]

plt.imshow(red_channel, cmap = 'hot')
plt.yticks([])
plt.xticks([])
#plt.title('Information in first PCA dimension')
#plt.savefig('First_PCA_dim.png', dpi = 200)
#plt.show()

"""Give PCA Statistic"""
print('explained variance per reduced dimension:', pca.explained_variance_ratio_)
print('cumulative explained variance:', np.cumsum(pca.explained_variance_ratio_))

df_var = pd.DataFrame(data = pca.explained_variance_ratio_, columns = ['explained variance'])
fig_1 = px.bar(df_var, y = 'explained variance' )
fig_1.update_layout(
    xaxis_title='PCA Dimension',
    yaxis_title="Relative Explained Variance",
    font=dict(
        size=18)
    )

#fig_1.write_image("Variance_per_dim.png")
#fig_1.show()

"""MinMax Scaling all channels"""
max_scaler = MinMaxScaler()

red_channel = max_scaler.fit_transform(red_channel.reshape(-1,1)).reshape(HEIGHT, WIDTH)
green_channel = max_scaler.fit_transform(green_channel.reshape(-1,1)).reshape(HEIGHT,WIDTH)
blue_channel = max_scaler.fit_transform(blue_channel.reshape(-1,1)).reshape(HEIGHT,WIDTH)


"""Stack to RGB image"""
RGB_image = np.dstack((red_channel,green_channel,blue_channel))
plt.imshow(RGB_image)
plt.xticks([])
plt.yticks([])
#plt.title('RGB image of first three PCA dimensions')
#plt.savefig('Three_PCA_dim.png', dpi = 200)
#plt.show()



PCA_df = pd.DataFrame()
PCA_df['label'] = flat_mask[:]

#PCA_df = pd.DataFrame(data = red_channel.reshape(-1), columns = ['dim1'])
PCA_df['dim1'] = red_channel.reshape(-1)
PCA_df['dim2'] = green_channel.reshape(-1)
PCA_df['dim3'] = blue_channel.reshape(-1)

print(PCA_df.head(5))
fig3d = px.scatter_3d(PCA_df, x = 'dim1', y = 'dim2', z = 'dim3', color = 'label', symbol = 'label', size = 'label' )
fig3d.update_layout(legend_orientation="h")
fig3d.show()

#dis.dis(f(2))

#red_channel = np.reshape(output_PCA.T[0], (HEIGHT, WIDTH))
#green_channel = output_PCA.T[1]
#blue_channel = output_PCA.T[2]

#print(red_channel.shape)

#plt.imshow(red_channel, cmap = 'gray' )
#plt.show()
