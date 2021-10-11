import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth

from Preprocessing.plant_transforms import image_train_transform, mask_train_transform
from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple
from Preprocessing.dataset_plants_binary import CustomDatasetBinary


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


def get_bandwidth(emb):
    flattened_embeddings = emb.reshape(emb.shape[0], -1).transpose()
    bandwidth = estimate_bandwidth(flattened_embeddings)
    return bandwidth


if __name__ == '__main__':
    HEIGHT, WIDTH = 150, 150

    directory = '/Users/luisa/Documents/BA_Thesis/Datasets for Multiple Instance Seg/CVPPP2017_instances/training/A1/'

    Plants = CustomDatasetMultiple(dir=directory,
                                   transform=None,
                                   image_transform=image_train_transform(HEIGHT, WIDTH),
                                   mask_transform=mask_train_transform(HEIGHT, WIDTH))

    Plants2 = CustomDatasetBinary(dir=directory,
                                  transform=None,
                                  image_transform=image_train_transform(HEIGHT, WIDTH),
                                  mask_transform=mask_train_transform(HEIGHT, WIDTH))

    # img_example, mask_example = Plants.__getitem__(5)
    img_example, mask_example = Plants2.__getitem__(7)

    image = img_example.unsqueeze(0)
    mask = mask_example  # want semantic mask instead of mask
    # loaded_model = torch.load('/Users/luisa/Documents/BA_Thesis/Embedding UNet/Code/saved_models/time_evolution/epoch-29.pt')
    loaded_model = torch.load(
        '/Users/luisa/Documents/BA_Thesis/Embedding UNet/Code/saved_models/pretraining/pretrain_epoch-2.pt')

    loaded_model.eval()

    embedding = loaded_model(image).squeeze(0).detach().numpy()

    bng = get_bandwidth(embedding)

    print('beginning clustering')
    result = cluster_ms(embedding, bandwidth=bng)
    # result = cluster_dbscan(embedding, 0.5, 1)
    print('estimates bandwidth:', bng)
    print('okay')

    plt.subplot(1, 3, 1)
    plt.title('Image')
    plt.imshow(np.array(img_example.permute(1, 2, 0)))

    plt.subplot(1, 3, 2)
    plt.title('Mask')
    plt.imshow(mask_example.permute(1, 2, 0))

    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    plt.imshow(result)

    plt.imshow(result)
    plt.show()
