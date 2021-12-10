import os
import random
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import torch
from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple, image_train_transform, mask_train_transform
from Postprocessing.dim_reduction import pca
from model import UNet_spoco, UNet_small, UNet_spoco_new
from utils_post import load_val_image


def visualization_train(DIM_PCA = None):
    HEIGHT, WIDTH = 512, 512
    OUT_CHANNELS = 32
    Epochs = 600

    rel_path = '~/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'
    directory = os.path.expanduser(rel_path)


    image, mask = load_val_image(height=HEIGHT, index=1)

    """loading trained model"""
    rel_model_path = '~/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/full_UNet/run-dim32-height512-epochs3000/epoch-{}-dim32-s512.pt'.format(Epochs)
    model_path = os.path.expanduser(rel_model_path)

    loaded_model = UNet_spoco_new(in_channels=3, out_channels=OUT_CHANNELS)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    """Forward pass to get embeddings"""
    embedding = loaded_model(image).squeeze(0)

    flat_embedding = embedding.detach().numpy().reshape((OUT_CHANNELS, -1))
    flat_mask = mask.reshape(-1)

    if DIM_PCA is not None:
        pca_output = pca(embedding.unsqueeze(0), output_dimensions=3).squeeze(0).view(DIM_PCA, HEIGHT, WIDTH)

        reduced_embedding = pca(embedding.unsqueeze(0), output_dimensions=3).squeeze(0).detach().numpy().reshape((DIM_PCA, -1))


        if DIM_PCA == 3:
            plt.imshow(pca_output.permute(1, 2, 0), cmap='Accent')
            plt.xticks(([]))
            plt.yticks(([]))
            plt.show()

        """Create DataFrame of reduced embedding (via PCA)"""
        pca_df = pd.DataFrame()
        pca_df['label'] = flat_mask[:]

        for i in range(0, DIM_PCA):
            pca_df['dim{}'.format(i + 1)] = reduced_embedding[i][:]

        pca_df['label'] = pca_df['label'] + 1

        fig_pca = px.scatter_3d(pca_df, x='dim1', y='dim2', z='dim3', color='label', symbol='label', size='label')
        fig_pca.show()


    unique_val = np.unique(flat_mask)

    """Create DataFrame of fully dimensional embeddings"""
    df = pd.DataFrame()
    df["label"] = flat_mask[:]

    for i in range(0, OUT_CHANNELS):
        df['dim{}'.format(i + 1)] = flat_embedding[i][:]

    # filter out background
    df_bg_free = df.query('label != 0.0')

    # Plot selected dimensions
    df = df.sort_values(by=['label'])
    df['label'] = df['label'] + 1


    #fig = px.scatter(df, x = 'dim1', y = 'dim2', color = 'label', size = 'label', symbol = 'label')

    fig = px.scatter(df, x='dim1', y='dim2', color=df["label"].astype(str), symbol='label',
                        title = 'Instance Pixel Embeddings shown in selected dimensions',
                        width = 1600, height = 800)

    fig.update_layout(legend_orientation="h")
    fig.show()


def make_video(folder):
    HEIGHT, WIDTH = 512, 512
    OUT_CHANNELS = 2

    number_pixel = 10000

    torch.manual_seed(0)
    random.seed(0)

    rel_path = '~/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'
    directory = os.path.expanduser(rel_path)

    Plants = CustomDatasetMultiple(dir=directory,
                                   transform=None,
                                   image_transform=image_train_transform(HEIGHT, WIDTH),
                                   mask_transform=mask_train_transform(HEIGHT, WIDTH))

    train_set, val_set, test_set = torch.utils.data.random_split(Plants, [80, 28, 20])

    img_example, mask_example = train_set.__getitem__(0)
    image = img_example.unsqueeze(0)
    mask = mask_example
    flat_mask = mask.reshape(-1)

    """loading trained models"""


    rel_model_path = '~/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/video/epoch-994-dim2-s512.pt'
    model_path = os.path.expanduser(rel_model_path)

    contents = sorted(os.listdir(folder))
    models = np.array(list(filter(lambda k: 'epoch' in k, contents)))
    loaded_models = np.empty(models.shape)
    print(models)

    df = pd.DataFrame()

    for i, m in enumerate(models):
        loaded_model = UNet_small(in_channels=3, out_channels=OUT_CHANNELS)
        loaded_model.load_state_dict(torch.load(folder + models[i]))
        loaded_model.eval()

        embedding = loaded_model(image).squeeze(0)
        flat_embedding = embedding.detach().numpy().reshape((OUT_CHANNELS, -1))

        df_i = pd.DataFrame()
        df_i["label"] = (flat_mask[:])
        df_i['epoch'] = i

        for i in range(0, OUT_CHANNELS):
            df_i['dim{}'.format(i + 1)] = flat_embedding[i][:]

        df = df.append(df_i)


    df['label'] = df['label'] + 1


    print(df.head(10))

    """Remove a random subset"""
    remove_n = int(512*512/10)
    drop_indices = np.random.choice(df.index, remove_n, replace=False)
    #df_subset = df.drop(drop_indices)
    df = df.drop(drop_indices)

    fig = px.scatter(df, x="dim1", y="dim2", animation_frame="epoch", symbol='label', color='label',
                     width=1600, height=1000,
                     title='Instance Embeddings',
                     range_x=[0, 20], range_y=[0, 20])

    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 0
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 3*(len(models))
    fig.update_layout(legend_orientation="h")

    #fig.show()
    fig.write_html("animation2D.html")

    # fig = px.scatter_3d(df_bg_free, x='dim1', y='dim2', z='dim6', color='label', symbol='label', size = 'label',
    # title = 'Instance Pixel Embeddings shown in selected dimensions',
    # width = 1200, height = 800)

    #fig = px.scatter(df, x='dim1', y='dim2', color='label', symbol='label', size='label',
                     #title='Instance Pixel Embeddings shown in selected dimensions',
                     #width=1600, height=800)


    # fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=0, x=1.1,
    #                                         ticks="outside"))
    # filtered_embedding = df_bg_free.drop('label', axis=1).to_numpy().reshape(-1, 16)


if __name__ == '__main__':
    visualization_train(DIM_PCA=3)
    #make_video('/Users/luisaneubauer/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/video/video_small_2/')


