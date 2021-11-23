import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import hdbscan
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth, AgglomerativeClustering

from Preprocessing.plant_transforms import image_train_transform, mask_train_transform
from Code.Preprocessing.dataset_plants_multiple import CustomDatasetMultiple
#from Code.Preprocessing.dataset_plants_binary import CustomDatasetBinary
from Code.model_from_spoco import UNet_spoco

def cluster(emb, clustering_alg, semantic_mask=None):
    output_shape = emb.shape[1:]
    # reshape numpy array (E, D, H, W) -> (E, D * H * W) and transpose -> (D * H * W, E)
    flattened_embeddings = emb.reshape(emb.shape[0], -1).transpose()

    result = np.zeros(flattened_embeddings.shape[0])

    if semantic_mask is not None:
        flattened_mask = semantic_mask.reshape(-1)
        assert flattened_mask.shape[0] == flattened_embeddings.shape[0]
    else:
        flattened_mask = np.ones(flattened_embeddings.shape[0])

    if flattened_mask.sum() == 0:
        # return zeros for empty masks
        return result.reshape(output_shape)

    # cluster only within the foreground mask
    clusters = clustering_alg.fit_predict(flattened_embeddings[flattened_mask == 1])
    # always increase the labels by 1 cause clustering results start from 0 and we may loose one object
    result[flattened_mask == 1] = clusters + 1

    return result.reshape(output_shape)


def cluster_dbscan(emb, eps, min_samples, semantic_mask=None):
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    return cluster(emb, clustering, semantic_mask)


def cluster_ms(emb, bandwidth, semantic_mask=None):
    clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    return cluster(emb, clustering, semantic_mask)

def cluster_agglo(emb, semantic_mask = None):
    clustering = AgglomerativeClustering()
    return cluster(emb, clustering, semantic_mask)

def cluster_hdbscan(emb, min_size, eps, min_samples=None, semantic_mask=None):
    clustering = hdbscan.HDBSCAN(min_cluster_size=min_size, cluster_selection_epsilon=eps, min_samples=min_samples)
    return cluster(emb, clustering, semantic_mask)

def get_bandwidth(emb):
    flattened_embeddings = emb.reshape(emb.shape[0], -1).transpose()
    bandwidth = estimate_bandwidth(flattened_embeddings)
    return bandwidth


if __name__ == '__main__':
    HEIGHT, WIDTH = 100, 100

    rel_path = '~/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'
    directory = os.path.expanduser(rel_path)

    Plants = CustomDatasetMultiple(dir=directory,
                                   transform=None,
                                   image_transform=image_train_transform(HEIGHT, WIDTH),
                                   mask_transform=mask_train_transform(HEIGHT, WIDTH))

    #Plants2 = CustomDatasetBinary(dir=directory,
     #                             transform=None,
      #                            image_transform=image_train_transform(HEIGHT, WIDTH),
       #                           mask_transform=mask_train_transform(HEIGHT, WIDTH))

    img_example, mask_example = Plants.__getitem__(2)

    image = img_example.unsqueeze(0)
    mask = mask_example  # want semantic mask instead of mask
    rel_model_path = '~/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/time_evolution/epoch-1000.pt'
    model_path = os.path.expanduser(rel_model_path)

    loaded_model = torch.load(model_path)

    loaded_model.eval()

    embedding = loaded_model(image).squeeze(0).detach().numpy()
    print('Forward Pass Done')

    #bng = get_bandwidth(embedding)
    #print('Bandwidth Estimation Done')
    #print(bng)

    print('Beginning Clustering')
    #result = np.array(cluster_ms(embedding, bandwidth=bng) - 1, np.int)  # labels start at 0
    n_min = 5
    epsilon = 0.2

    #result = cluster_agglo(embedding)

    #result = cluster_dbscan(embedding, n_min, epsilon)
    print(embedding.shape)
    result = cluster_hdbscan(embedding, n_min, epsilon)
    print('Number of Instances Detected:', np.unique(result))
    print('Number of Instances in Ground Truth:', np.unique(mask_example))
    #print('estimates bandwidth:', bng)
    print('Clustering Done')

    fig = plt.figure(figsize=(16,12))
    plt.title(r'HDBSCAN with $n_m = {}$ and $\epsilon = {}$'.format(n_min, epsilon))
    plt.subplot(1, 3, 1)
    plt.title('Image', size = 'large')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.array(img_example.permute(1, 2, 0)))

    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Mask', size = 'large')
    plt.imshow(mask_example.permute(1, 2, 0), cmap = 'Spectral', interpolation = 'nearest')

    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask', size = 'large')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(result , cmap = 'Spectral', interpolation = 'nearest')

    fig.savefig('Segmentation.png', dpi = 200)

    plt.show()

    mask_example = np.array(mask_example.detach().numpy(), np.int)


