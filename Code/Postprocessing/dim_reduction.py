import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple, mask_train_transform, image_train_transform
from model import UNet_small, UNet_spoco_new


def TSNE_with_scaling(embedding, mask):
    """Flatten Image"""
    flat_embedding = embedding.reshape((16, -1))

    """1D to 2D: check if transformation works"""
    embedding = flat_embedding.reshape((16, HEIGHT, WIDTH))

    """Defining scaler and dimension reduction technique"""
    scaler = StandardScaler()
    tsne = TSNE(n_components=DIM_RED)

    """Preparing input for Dimension Reduction technique"""
    input = scaler.fit_transform(flat_embedding)
    print(input.shape)

    """Check if image still looks nice"""
    check = input.reshape((16, HEIGHT, WIDTH))
    plt.imshow(check[1], cmap='hot')
    plt.xticks([])
    plt.yticks([])
    # plt.title('Information in first embedding dimension after rescaling')
    plt.savefig('First_TSNE.png', dpi=200)
    plt.show()

    """DO TSNE"""
    output = tsne.fit_transform(input.T).T

    image_TSNE = output.reshape((DIM_RED, HEIGHT, WIDTH))
    print(image_TSNE.shape)

    red_channel = image_TSNE[0]
    green_channel = image_TSNE[1]
    blue_channel = image_TSNE[2]

    plt.imshow(red_channel, cmap='hot')
    plt.title('Information in first TSNE dimension')
    plt.show()


def PCA_with_Scaling(embedding, mask):
    HEIGHT = embedding.size[-1]

    flat_embedding = embedding.reshape((16, -1))
    flat_mask = mask.reshape(-1)

    """1D to 2D: check if transformation works"""
    embedding = flat_embedding.reshape((16, HEIGHT, WIDTH))
    # plt.imshow(embedding[1], cmap = 'gray')
    # plt.show()

    """Defining Scaler and Dimension Reduction Technique"""

    scaler = StandardScaler()
    pca = PCA(n_components=PCA_dim)

    """Fitting Scaler"""
    input_for_PCA = scaler.fit_transform(flat_embedding)
    print(input_for_PCA.shape)

    """Check if image still looks nice"""
    check = input_for_PCA.reshape((16, HEIGHT, WIDTH))
    # plt.imshow(check[1], cmap = 'gray')
    # plt.show()

    """Do PCA"""
    output_PCA = pca.fit_transform(input_for_PCA.T).T

    image_PCA = output_PCA.reshape((PCA_dim, HEIGHT, WIDTH))

    red_channel = image_PCA[0]
    green_channel = image_PCA[1]
    blue_channel = image_PCA[2]

    plt.imshow(red_channel, cmap='hot')
    plt.yticks([])
    plt.xticks([])
    # plt.title('Information in first PCA dimension')
    # plt.savefig('First_PCA_dim.png', dpi = 200)
    # plt.show()

    """Give PCA Statistic"""
    print('explained variance per reduced dimension:', pca.explained_variance_ratio_)
    print('cumulative explained variance:', np.cumsum(pca.explained_variance_ratio_))

    df_var = pd.DataFrame(data=pca.explained_variance_ratio_, columns=['explained variance'])
    fig_1 = px.bar(df_var, y='explained variance')
    fig_1.update_layout(
        xaxis_title='PCA Dimension',
        yaxis_title="Relative Explained Variance",
        font=dict(
            size=18)
    )

    # fig_1.write_image("Variance_per_dim.png")
    # fig_1.show()

    """MinMax Scaling all channels"""
    max_scaler = MinMaxScaler()

    red_channel = max_scaler.fit_transform(red_channel.reshape(-1, 1)).reshape(HEIGHT, WIDTH)
    green_channel = max_scaler.fit_transform(green_channel.reshape(-1, 1)).reshape(HEIGHT, WIDTH)
    blue_channel = max_scaler.fit_transform(blue_channel.reshape(-1, 1)).reshape(HEIGHT, WIDTH)

    """Stack to RGB image"""
    RGB_image = np.dstack((red_channel, green_channel, blue_channel))
    plt.imshow(RGB_image)
    plt.xticks([])
    plt.yticks([])
    # plt.title('RGB image of first three PCA dimensions')
    # plt.savefig('three_PCA_dim.png', dpi = 200)
    # plt.show()

    PCA_df = pd.DataFrame()
    PCA_df['label'] = flat_mask[:]

    # PCA_df = pd.DataFrame(data = red_channel.reshape(-1), columns = ['dim1'])
    PCA_df['dim1'] = red_channel.reshape(-1)
    PCA_df['dim2'] = green_channel.reshape(-1)
    PCA_df['dim3'] = blue_channel.reshape(-1)

    print(PCA_df.head(5))
    fig3d = px.scatter_3d(PCA_df, x='dim1', y='dim2', z='dim3', color='label', symbol='label', size='label')
    fig3d.update_layout(legend_orientation="h")
    fig3d.show()

    # dis.dis(f(2))

    # red_channel = np.reshape(output_PCA.T[0], (HEIGHT, WIDTH))
    # green_channel = output_PCA.T[1]
    # blue_channel = output_PCA.T[2]

    # print(red_channel.shape)

    # plt.imshow(red_channel, cmap = 'gray' )
    # plt.show()


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
    from utils import load_image_mask
    E_DIM = 8
    HEIGHT, WIDTH = 512, 512
    PCA_dim = 3
    EPOCH = 2000

    model_path = '/Users/luisaneubauer/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/full_UNet/run-dim{}-height512-epochs3000/epoch-{}-dim{}-s512.pt'.format(E_DIM, EPOCH, E_DIM)
    loaded_model = UNet_spoco_new(in_channels=3, out_channels=E_DIM)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()


    directory = '/Users/luisaneubauer/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'

    img_example, mask_example = load_image_mask(height=HEIGHT, index = 8, mode='train')

    image = img_example #.unsqueeze(0)
    mask = mask_example
    flat_mask = mask.view(HEIGHT*WIDTH)

    embedding = loaded_model(image)
    print(embedding.shape)
    pca_output = pca(embedding, output_dimensions=PCA_dim).squeeze(0).view(PCA_dim, HEIGHT*WIDTH)

    flat_embedding = embedding.reshape((E_DIM, -1))

    #np.savetxt('embedding_{}_{}.csv'.format(PCA_dim, HEIGHT), flat_embedding.detach().numpy(), delimiter=",")
    #remove_n = int(512 * 512 / 10)

    #drop_indices = np.random.choice(len(flat_mask), remove_n, replace=False)
    # df_subset = df.drop(drop_indices)
    #flat_mask = np.delete(flat_mask, drop_indices)#df.drop(drop_indices)
    #print(flat_mask.shape)
    #pca_output = np.delete(pca_output[:], drop_indices)


    #fig = px.scatter_3d(x = pca_output[0,:], y = pca_output[1,:], z = pca_output[2,:], color=flat_mask, size = flat_mask+1, symbol=flat_mask)
    #fig.show()
    normal_output = embedding.squeeze()[0].detach().numpy() #.permute(1,2,0).detach().numpy()
    print(normal_output.shape)
    plt.yticks([])
    plt.xticks([])
    plt.imshow(normal_output, cmap = 'gray')
    plt.savefig('First_DIM.png', dpi=200)
    plt.show()


    pca_output = pca_output.view(PCA_dim, HEIGHT, WIDTH)
    print(pca_output.shape)
    plt.yticks([])
    plt.xticks([])

    if pca_output.shape[0] > 1:
        plt.imshow(pca_output.squeeze().permute(1,2,0), cmap='gray') #.permute(1, 2, 0))

    else:
        plt.imshow(pca_output.squeeze(), cmap='gray')

    plt.savefig('PCA{}_to{}.png'.format(E_DIM, PCA_dim), dpi = 200)
    plt.show()



if __name__ == '__main__':
    test()
