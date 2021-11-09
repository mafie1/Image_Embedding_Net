import plotly.express as px
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple
from Preprocessing.dataset_plants_binary import CustomDatasetBinary
from Preprocessing.plant_transforms import image_train_transform, mask_train_transform

HEIGHT, WIDTH = 128, 128

directory = '/Users/luisa/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'


def visualization_train():

    Plants = CustomDatasetMultiple(dir=directory,
                                   transform=None,
                                   image_transform=image_train_transform(HEIGHT, WIDTH),
                                   mask_transform=mask_train_transform(HEIGHT, WIDTH))

    img_example, mask_example = Plants.__getitem__(1)
    image = img_example.unsqueeze(0)
    mask = mask_example

    """training on multiple instances"""
    loaded_model = torch.load('/Users/luisa/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/long_runs/epoch-100.pt')
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
    print(flat_embedding.shape)
    #print(df_bg_free.keys())
    #print(df_bg_free.head(5))


    #print(df.sum())
    #print(df_bg_free.std())
    #print(df_zeroed.head(10))

    # Plot selected dimensions

    #fig = px.scatter(df_bg_free, x = 'dim1', y = 'dim2', color = 'label', marginal_x="rug", marginal_y="rug", title='background free')
    #fig = px.scatter_3d(df_bg_free, x = 'dim1', y = 'dim2', z = 'dim3', color = 'label', size = 'label', symbol = 'label')
    df['label'] = df['label']+1
    #fig = px.scatter_3d(df_bg_free, x='dim1', y='dim2', z='dim6', color='label', symbol='label', size = 'label',
                       # title = 'Instance Pixel Embeddings shown in selected dimensions',
                        #width = 1200, height = 800)

    fig = px.scatter(df, x='dim2', y='dim3', color='label', symbol='label', size = 'label',
                        title = 'Instance Pixel Embeddings shown in selected dimensions',
                        width = 1200, height = 800)
    fig.update_layout(legend_orientation="h")
    #fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=0, x=1.1,
     #                                         ticks="outside"))
    fig.show()

    #filtered_embedding = df_bg_free.drop('label', axis=1).to_numpy().reshape(-1, 16)

    #Preparing Data for PCA

"""
    data = df.drop('label', axis = 1)
    print(data.head(5))

    scaler = StandardScaler()
    scaler.fit(data)

    scaled_data =scaler.transform(data)

    pca = PCA(n_components = 3)
    pca.fit(scaled_data)

    x_pca = pca.transform(scaled_data)
    print(scaled_data.shape, x_pca.shape)

    x_pca = x_pca.reshape((3,-1))

    x_pca_df = pd.DataFrame()
    x_pca_df['label'] = df.label
    x_pca_df["PCA dim 1"] = x_pca[0][:]
    x_pca_df['PCA dim 2'] = x_pca[1][:]
    x_pca_df['PCA dim 3'] = x_pca[2][:]

    print(x_pca_df.head(5))

    fig1 = px.scatter_3d(x_pca_df, x='PCA dim 1', y='PCA dim 2', z='PCA dim 3', color='label', symbol = 'label')
    fig1.show()

"""
    #print(flat_embedding.shape)

    #PCA of Filtered Labels (without background)
    #input_for_PCA = flat_embedding.transpose()
    #scaler = StandardScaler()

    #input_for_PCA = scaler.fit_transform(input_for_PCA).transpose()
    #scaled_instances = np.array([input_for_PCA * masks[i].reshape((16, -1)) for i in values])
    #print(input_for_PCA.shape)

    #print(filtered_embedding.shape)

    #pca = PCA(n_components=3)
    #pca.fit(input_for_PCA)
    #X = pca.transform(input_for_PCA).reshape(3, -1)

    #print('Variance Explained:', pca.explained_variance_ratio_)
    #print('The singular values are: ', pca.singular_values_)

    #X = pca.fit_transform(filtered_embedding).reshape(3, -1)
    #print('PCA done')

    #df_PCA = pd.DataFrame()
    #df_PCA['label'] = df_bg_free['label']
    #df_PCA["PCA_dim_1"] = X[0][:]
    #df_PCA["PCA_dim_2"] = X[1][:]
    #df_PCA["PCA_dim_3"] = X[2][:]

    #print(pca.singular_values_)

    #TSNE of Filtered Labels

    #tsne = TSNE(n_components = 3, n_iter=250)
    #X = tsne.fit_transform(filtered_embedding).reshape(3, -1)
    #print('TSNE done')
    #print(X.shape)


    #df_TSNE = pd.DataFrame()
    #df_TSNE["label"] = df_bg_free['label']
    #df_TSNE["TSNE_dim_1"] = X[0][:]
    #df_TSNE["TSNE_dim_2"] = X[1][:]
    #df_TSNE["TSNE_dim_3"] = X[2][:]

    #print(df_TSNE.head(5))


    #fig1 = px.scatter_3d(df_PCA, x = 'PCA_dim_1', y = 'PCA_dim_2', z = 'PCA_dim_3', color = 'label')
    # fig2 = px.scatter_3d(df_TSNE, x = 'TSNE_dim_1', y = 'TSNE_dim_2', z = 'TSNE_dim_3', color = 'label')

    #fig1.show()


if __name__ == '__main__':
    #visualization_pretrain()
    visualization_train()


