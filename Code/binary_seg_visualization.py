import torch
import torchvision.transforms as TF

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import numpy as np
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from torch.autograd import Variable


def single_image_loader(image_path):
    """loads image, returns cuda tensor"""
    imsize = 300
    loader = TF.Compose([TF.Resize(imsize), TF.ToTensor()])
    #loader = TF.Compose([TF.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image = loader(image).float()
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image  # assumes that you're using GPU

def single_mask_loader(mask_path):
    imsize = 300
    loader = TF.Compose([TF.Resize(imsize), TF.ToTensor()])
    #loader = TF.Compose([TF.ToTensor()])
    mask = Image.open(mask_path).convert('L')
    mask = loader(mask)
    return mask


loaded_model = torch.load('/Users/luisa/Documents/BA_Thesis/Embedding UNet/Code/saved_models/binary_seg/epoch-4.pt')
loaded_model.eval()

# multi-instance segmentation
#img_path = '/Users/luisa/Documents/BA_Thesis/Datasets for Multiple Instance Seg/CVPPP2017_instances/training/A1/plant018_rgb.png'
#mask_path = '/Users/luisa/Documents/BA_Thesis/Datasets for Multiple Instance Seg/CVPPP2017_instances/training/A1/plant018_label.png'

# binary segmentation
img_path = '/Users/luisa/Documents/BA_Thesis/CVPPP2015_LCC_training_data/A1/plant001_rgb.png'
mask_path = '/Users/luisa/Documents/BA_Thesis/CVPPP2015_LCC_training_data/A1/plant001_fg.png'


image = single_image_loader(img_path)
mask = single_mask_loader(mask_path)

#mask_bg = np.where(mask == 0, 1, 0).squeeze()
#instance = np.where(mask != 0, 1,0).squeeze()

embedding = loaded_model(image).squeeze(0)
flat_embedding = embedding.detach().numpy().reshape((-1, 16))
flat_mask = mask.reshape(-1)
mask_4 = torch.stack((flat_mask, flat_mask, flat_mask, flat_mask))
mask_16 = torch.vstack((mask_4, mask_4, mask_4, mask_4)).reshape((-1,16)).detach().numpy()


instance_0 = (flat_embedding * mask_16).reshape((16, -1))
background = ((1 - mask_16)*(-1)*flat_embedding).reshape((16, -1))


plt.figure(figsize = (10,10))
plt.title('Position of instances in the embedding space (ground truth, binary segmentation)')

"""2D Plot"""
#plt.xlabel('dimension 1')
#plt.ylabel('dimension 2')
#plt.grid()
#plt.scatter(instance_0[0], instance_0[1], label = 'embedding of foreground')
#plt.scatter(background[0], background[1], label = 'background')
#plt.legend()

"""3D Plot"""
#ax = plt.axes(projection='3d')

#ax.set_xlabel('Dimension 1')
#ax.set_ylabel('Dimension 2')
#ax.set_zlabel('Dimension 5')

#ax.scatter3D(instance_0[0], instance_0[1], instance_0[6], label = 'foreground')
#ax.scatter3D(background[0], background[1], background[6], label = 'background')

#plt.legend()
#plt.show()

#flat_embedding = flat_embedding.transpose()

pca = PCA(n_components=3)
tsne = TSNE(n_components=3, n_iter=250)

#pca.fit(flat_embedding)
#X = pca.transform(flat_embedding)
#mask_PCA = pca.transform(mask_16)

#instance_0 = (X * mask_PCA).reshape((3, -1))
#background = ((1 - mask_PCA)*(-1)*X).reshape((3, -1))

X = tsne.fit_transform(flat_embedding)
print('done')
mask_TSNE = tsne.fit_transform(mask_16)

instance_0 = (X * mask_TSNE).reshape((3, -1))
background = ((1 - mask_TSNE)*(-1)*X).reshape((3, -1))


X = X.reshape((3, -1))

plt.title('TSNE new of embedding')
ax = plt.axes(projection='3d')

ax.set_xlabel('TSNE - Dimension 1')
ax.set_ylabel('TSNE - Dimension 2')
ax.set_zlabel('TSNE - Dimension 3')

ax.scatter3D(instance_0[0], instance_0[1], instance_0[2], label = 'foreground')
ax.scatter3D(background[0], background[1], background[2], label = 'background')
#ax.scatter3D(X[0], X[1], X[2])
plt.show()



#plt.figure(figsize=(10, 10))
#plt.scatter(X[0], X[1])
#plt.show()

