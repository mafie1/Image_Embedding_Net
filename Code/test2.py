import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def pca_project(embeddings):
    """
    Project embeddings into 3-dim RGB space for visualization purposes
    Args:
        embeddings: ExSpatial embedding tensor
    Returns:
        RGB image
    """
    assert embeddings.ndim == 3
    embeddings = embeddings.detach().numpy()
    # reshape (C, H, W) -> (C, H * W) and transpose
    flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).transpose() #numpy transpose method
    print(flattened_embeddings.shape)
    # init PCA with 3 principal components: one for each RGB channel
    pca = PCA(n_components=3)
    # fit the model with embeddings and apply the dimensionality reduction
    flattened_embeddings = pca.fit_transform(flattened_embeddings)
    # reshape back to original
    shape = list(embeddings.shape)
    shape[0] = 3
    img = flattened_embeddings.transpose().reshape(shape)
    # normalize to [0, 255]
    img = 255 * (img - np.min(img)) / np.ptp(img)
    return img.astype('uint8')


def mask_to_integer_mask(mask):
    uniques = np.unique(mask)
    integers = np.linspace(0, len(uniques) - 1, len(uniques))

    for i in range(0, len(uniques)):
        mask = np.where(mask == uniques[i], integers[i], mask)
    mask = np.array(mask, dtype=np.int64)
    return mask

if __name__ == '__main__':
    image = np.zeros(500, 500)
    fig = plt.imshow(image)
    plt.savefig(fig)