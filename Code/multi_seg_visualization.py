import torch
import matplotlib.pyplot as plt
import numpy as np

from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple
from Preprocessing.plant_transforms import image_train_transform, mask_train_transform

HEIGHT, WIDTH = 256, 256
directory = '/Users/luisa/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'

Plants = CustomDatasetMultiple(dir = directory,
                               transform= None,
                               image_transform= image_train_transform(HEIGHT, WIDTH),
                               mask_transform= mask_train_transform(HEIGHT, WIDTH))

img_example, mask_example = Plants.__getitem__(5)

image = img_example.unsqueeze(0)
mask = mask_example

#plt.subplot(1, 2, 1)
#plt.title('Image')
#plt.imshow(np.array(img_example.permute(1, 2, 0)))
#plt.subplot(1, 2, 2)
#plt.title('Mask')
#plt.imshow(mask_example.permute(1, 2, 0))
#plt.show()

loaded_model = torch.load('/Users/luisa/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/long_runs/epoch-100.pt')
loaded_model.eval()



embedding = loaded_model(image).squeeze(0)
flat_embedding = embedding.detach().numpy().reshape((16, -1))
flat_mask = mask.reshape(-1)


values, counts = np.unique(flat_mask, return_counts=True)
values = np.array(values, dtype=int)


dimension_a = 0
dimension_b = 1
dimension_c = 2

dimension_d = 13
dimension_e = 14
dimension_f = 15

def create_masks_for_instance_N(mask, N):
    flat_mask = mask.reshape(-1)
    C = len(np.unique(flat_mask))
    assert N <= C

    mask_4 = torch.stack((flat_mask, flat_mask, flat_mask, flat_mask))
    mask_16 = torch.vstack((mask_4, mask_4, mask_4, mask_4)).reshape((16, -1)).detach().numpy()

    example_mask_N = np.where(mask_16 == np.unique(mask_16)[N], 1, 0).reshape((16, -1))
    return example_mask_N

masks = np.array([create_masks_for_instance_N(mask, i) for i in values])
instances = np.array([flat_embedding*masks[i].reshape((16, -1)) for i in values])

fig = plt.figure(figsize = (20,10))

plt.title('Show first three dimensions of embedding')

ax = fig.add_subplot(1, 2, 1, projection='3d')
for i in values[1:]:
    ax.scatter3D(instances[i][dimension_a], instances[i][dimension_b], instances[i][dimension_c], label = 'Instance {}'.format(i))

ax = fig.add_subplot(1, 2, 2, projection='3d')
for i in values[1:]:
    ax.scatter3D(instances[i][dimension_d], instances[i][dimension_e], instances[i][dimension_f], label = 'Instance {}'.format(i))

plt.show()


#tsne = TSNE(n_components = 3, n_iter=250)
#X = tsne.fit_transform(flat_embedding.transpose()).reshape(3, -1)

#print('TSNE done')
#print(flat_mask.shape)
#print(X.shape)



#df = pd.DataFrame()
#df["y"] = flat_mask[:]
#df["TSNE_dim_1"] = X[0][:]
#df["TSNE_dim_2"] = X[1][:]
#df["TSNE_dim_3"] = X[2][:]

#print(df.head(5))

#df = df.drop('y' == 0.0)

#plt.figure(figsize = (15, 15))
#sns.scatterplot(x="TSNE_dim_1", y="TSNE_dim_3", hue=df.y.tolist(),
               # palette=sns.color_palette("hls", C ),
                #data=df).set(title="data T-SNE projection")
#plt.show()





#plt.title('TSNE new of embedding')
#ax = plt.axes(projection='3d')

#ax.set_xlabel('TSNE - Dimension 1')
#ax.set_ylabel('TSNE - Dimension 2')
#ax.set_zlabel('TSNE - Dimension 3')

#ax.scatter3D(X[0], X[1], X[2])
#plt.show()



#clustering = OPTICS(min_samples = 5).fit(flat_embedding)
#labels = clustering.labels_


