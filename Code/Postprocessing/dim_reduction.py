import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from copy import deepcopy
from utils import save_embedding
from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple, mask_train_transform, image_train_transform


def pca(embedding, output_dimensions=3, reference=None, center_data=False, return_pca_objects=False):
    # embedding shape: first two dimensions corresponde to batchsize and embedding dim, so
    # shape should be (B, E, H, W) or (B, E, D, H, W).
    _pca = PCA(n_components=output_dimensions)
    # reshape embedding
    output_shape = list(embedding.shape)
    output_shape[1] = output_dimensions
    flat_embedding = embedding.detach().numpy().reshape(embedding.shape[0], embedding.shape[1], -1)
    flat_embedding = flat_embedding.transpose((0, 2, 1))
    if reference is not None:
        # assert reference.shape[:2] == embedding.shape[:2]
        flat_reference = reference.detach().numpy().reshape(reference.shape[0], reference.shape[1], -1)\
            .transpose((0, 2, 1))
    else:
        flat_reference = flat_embedding

    if center_data:
        means = np.mean(flat_reference, axis=0, keepdims=True)
        flat_reference -= means
        flat_embedding -= means

    pca_output = []
    pca_objects = []
    for flat_reference, flat_image in zip(flat_reference, flat_embedding):
        # fit PCA to array of shape (n_samples, n_features)..
        _pca.fit(flat_reference)
        # ..and apply to input data
        pca_output.append(_pca.transform(flat_image))
        if return_pca_objects:
            pca_objects.append(deepcopy(_pca))
    transformed = torch.stack([torch.from_numpy(x.T) for x in pca_output]).reshape(output_shape)
    if not return_pca_objects:
        return transformed
    else:
        return transformed, pca_objects




def test():
    loaded_model = torch.load(
        '/Users/luisaneubauer/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/time_evolution/epoch-1000.pt')
    loaded_model.eval()

    HEIGHT, WIDTH = 100, 100
    PCA_dim = 3

    directory = '/Users/luisaneubauer/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'

    Plants = CustomDatasetMultiple(dir=directory,
                                   transform=None,
                                   image_transform=image_train_transform(HEIGHT, WIDTH),
                                   mask_transform=mask_train_transform(HEIGHT, WIDTH))

    img_example, mask_example = Plants.__getitem__(2)

    image = img_example.unsqueeze(0)
    mask = mask_example
    embedding = loaded_model(image)

    print(pca(embedding).shape)

    pca_output = pca(embedding, output_dimensions=PCA_dim).squeeze(0).view(PCA_dim,HEIGHT,WIDTH)

    flat_embedding = embedding.reshape((16, -1))

    np.savetxt('embedding_{}_{}.csv'.format(PCA_dim, HEIGHT), flat_embedding.detach().numpy(), delimiter=",")
    print(flat_embedding.shape)
    plt.imshow(pca_output.squeeze().permute(1,2,0))
    plt.show()



if __name__ == '__main__':
    test()
