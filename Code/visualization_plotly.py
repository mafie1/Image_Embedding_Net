import plotly.express as px
import numpy as np
import pandas as pd
import torch
from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple
from Preprocessing.dataset_plants_binary import CustomDatasetBinary
from Preprocessing.plant_transforms import image_train_transform, mask_train_transform

HEIGHT, WIDTH = 50, 50

directory = '/Users/luisa/Documents/BA_Thesis/Datasets for Multiple Instance Seg/CVPPP2017_instances/training/A1/'



def visualization_pretrain():
    Plants = CustomDatasetBinary(dir=directory,
                                   transform=None,
                                   image_transform=image_train_transform(HEIGHT, WIDTH),
                                   mask_transform=mask_train_transform(HEIGHT, WIDTH))

    img_example, mask_example = Plants.__getitem__(10)
    image = img_example.unsqueeze(0)
    mask = mask_example

    """pre_training on foreground/background"""
    loaded_model = torch.load('/Users/luisa/Documents/BA_Thesis/Image Embedding Net/Code/saved_models/pretraining/pretrain_epoch-20.pt')
    loaded_model.eval()

    embedding = loaded_model(image).squeeze(0)
    flat_embedding = embedding.detach().numpy().reshape((16, -1))
    flat_mask = mask.reshape(-1)

    unique_val = np.unique(flat_mask)
    print('The unique values in the image mask are:', unique_val)

    df = pd.DataFrame()
    df["label"] = flat_mask[:]

    for i in range(0, 16):
        df['dim{}'.format(i + 1)] = flat_embedding[i][:]


    # filter out background
    df_bg_free = df.query('label != 0.0')

    print('Background Free Dataframe:')
    print(df_bg_free.head(5))

    # drop zero.dimensions
    df_zeroed = df.drop('dim1', axis=1).drop('dim2', axis=1).drop('dim3', axis=1)

    print(df_zeroed.head(5))

    # Plot selected dimensions

    #fig = px.scatter(df, x = 'dim1', y = 'dim2',
     #               color = 'label',
      #              symbol = 'label',
       #             marginal_x="rug",
        #            marginal_y="rug",
         #           title='all labels')

    fig = px.scatter_3d(df, x='dim4', y='dim5', z='dim6', color='label', symbol='label')
    fig.show()

    filtered_embedding = df_bg_free.drop('label', axis=1).to_numpy().reshape(-1, 16)



def visualization_train():

    Plants = CustomDatasetMultiple(dir=directory,
                                   transform=None,
                                   image_transform=image_train_transform(HEIGHT, WIDTH),
                                   mask_transform=mask_train_transform(HEIGHT, WIDTH))

    img_example, mask_example = Plants.__getitem__(10)
    image = img_example.unsqueeze(0)
    mask = mask_example

    """training on multiple instances"""
    loaded_model = torch.load('/Users/luisa/Documents/BA_Thesis/Image Embedding Net/Code/saved_models/time_evolution/epoch-29.pt')
    loaded_model.eval()

    embedding = loaded_model(image).squeeze(0)
    flat_embedding = embedding.detach().numpy().reshape((16, -1))
    flat_mask = mask.reshape(-1)

    unique_val = np.unique(flat_mask)

    df = pd.DataFrame()
    df["label"] = flat_mask[:]

    for i in range(0, 16):
        df['dim{}'.format(i + 1)] = flat_embedding[i][:]

    # filter out background
    df_bg_free = df.query('label != 0.0')

    print(df_bg_free.head(5))

    # drop zero.dimension

    df_zeroed = df.drop('dim15', axis=1).drop('dim16', axis=1).drop('dim12', axis=1).drop('dim1', axis=1).drop('dim2',
                                                                                                               axis=1)

    print(df_zeroed.head(10))

    # Plot selected dimensions

    # fig = px.scatter_(df, x = 'dim1', y = 'dim5', color = 'label', marginal_x="rug", marginal_y="rug", title='background free')
    fig = px.scatter_3d(df_bg_free, x = 'dim3', y = 'dim10', z = 'dim6', color = 'label', size = 'label', symbol = 'label')
    #fig = px.scatter_3d(df, x='dim3', y='dim10', z='dim6', color='label', symbol='label')
    # fig.show()

    fig.show()

    filtered_embedding = df_bg_free.drop('label', axis=1).to_numpy().reshape(-1, 16)

    """
    #PCA of Filtered Labels (without background)

    pca = PCA(n_components=3)
    pca.fit(filtered_embedding)
    X = pca.transform(filtered_embedding).reshape(3, -1)

    print('Variance Explained:', pca.explained_variance_ratio_)
    print('The singular values are: ', pca.singular_values_)

    X = pca.fit_transform(filtered_embedding).reshape(3, -1)
    print('PCA done')

    df_PCA = pd.DataFrame()
    df_PCA['label'] = df_bg_free['label']
    df_PCA["PCA_dim_1"] = X[0][:]
    df_PCA["PCA_dim_2"] = X[1][:]
    df_PCA["PCA_dim_3"] = X[2][:]

    print(pca.singular_values_)

    #TSNE of Filtered Labels

    tsne = TSNE(n_components = 3, n_iter=250)
    X = tsne.fit_transform(filtered_embedding).reshape(3, -1)
    print('TSNE done')
    print(X.shape)


    df_TSNE = pd.DataFrame()
    df_TSNE["label"] = df_bg_free['label']
    df_TSNE["TSNE_dim_1"] = X[0][:]
    df_TSNE["TSNE_dim_2"] = X[1][:]
    df_TSNE["TSNE_dim_3"] = X[2][:]

    print(df_TSNE.head(5))
    """

    # fig1 = px.scatter_3d(df_PCA, x = 'PCA_dim_1', y = 'PCA_dim_2', z = 'PCA_dim_3', color = 'label')
    # fig2 = px.scatter_3d(df_TSNE, x = 'TSNE_dim_1', y = 'TSNE_dim_2', z = 'TSNE_dim_3', color = 'label')

    # fig1.show()


if __name__ == '__main__':
    #visualization_pretrain()
    visualization_train()


