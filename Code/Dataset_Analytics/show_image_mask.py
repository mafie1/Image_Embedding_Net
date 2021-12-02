import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import hdbscan

from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth, AgglomerativeClustering
from Code.Preprocessing.dataset_plants_multiple import CustomDatasetMultiple, image_train_transform, mask_train_transform
from Code.Preprocessing.dataset_plants_binary import CustomDatasetBinary


HEIGHT, WIDTH = 512, 512
rel_path = '~/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'
directory = os.path.expanduser(rel_path)
print(directory)
Plants = CustomDatasetMultiple(dir=directory,
                                transform=None,
                                image_transform=image_train_transform(HEIGHT, WIDTH),
                                mask_transform=mask_train_transform(HEIGHT, WIDTH))

Plants2 = CustomDatasetBinary(dir=directory,
                                transform=None,
                                image_transform=image_train_transform(HEIGHT, WIDTH),
                                mask_transform=mask_train_transform(HEIGHT, WIDTH))

img_example, mask_example = Plants.__getitem__(5)
binary_mask_example = Plants2.__getitem__(5)[1]

fig = plt.figure(figsize = (12, 4))
plt.xticks([])
plt.yticks([])
plt.axis('off')

ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.set_title('Image')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.axis('off')
ax1.imshow(np.array(img_example.permute(1, 2, 0)))


#plt.title('Image')
#plt.imshow(np.array(img_example.permute(1, 2, 0)))

#plt.subplot(1, 3, 2)
ax2.set_title('Mask')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.imshow(mask_example.permute(1, 2, 0)) #, cmap = 'tab10'

#plt.subplot(1, 3, 3)
ax3.set_title('Binary Mask')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.imshow(binary_mask_example.squeeze(), cmap = 'gray')

plt.savefig('Image_Overview.png', dpi=200)
plt.show()