import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import hdbscan
from tqdm import tqdm
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth, AgglomerativeClustering
from metrics import counting_score, get_SBD
from Code.Preprocessing.dataset_plants_multiple import CustomDatasetMultiple, image_train_transform, mask_train_transform
from Code.model import UNet_spoco, UNet_small
import csv


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
    clustering = AgglomerativeClustering(n_clusters=5)
    return cluster(emb, clustering, semantic_mask)


def cluster_hdbscan(emb, min_size, eps, min_samples=None, semantic_mask=None, metric = 'l2'):
    """For hsbscan the optimal parameters are in the range of:
    - for 200x200 images: min_size = 25 , epsilon = 0.4-0.5
    - for 400x400 images: min_size = 140, epsilon = 0.4-0.5
    increase both of 'l1' metric is used

    usefull metrics are: 'l1' (manhattan), 'l2' (euclidean)
    """
    clustering = hdbscan.HDBSCAN(min_cluster_size=min_size, cluster_selection_epsilon=eps, min_samples=min_samples, metric = metric)
    return cluster(emb, clustering, semantic_mask)

def get_bandwidth(emb):
    flattened_embeddings = emb.reshape(emb.shape[0], -1).transpose()
    bandwidth = estimate_bandwidth(flattened_embeddings)
    return bandwidth


def cluster_single_image(save = None, index = 3, n_min = 200, epsilon = 0.5, plot = None):
    HEIGHT, WIDTH = 512, 512

    rel_path = '~/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'
    directory = os.path.expanduser(rel_path)

    Plants = CustomDatasetMultiple(dir=directory,
                                   transform=None,
                                   image_transform=image_train_transform(HEIGHT, WIDTH),
                                   mask_transform=mask_train_transform(HEIGHT, WIDTH))

    torch.manual_seed(0)
    random.seed(0)
    train_set, val_set, test_set = torch.utils.data.random_split(Plants, [80, 28, 20])

    img_example, mask_example = val_set.__getitem__(index)

    image = img_example.unsqueeze(0)
    mask = mask_example  # want semantic mask instead of mask

    rel_model_path = '~/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/video/video_small/epoch-900-dim2-s512.pt'
    model_path = os.path.expanduser(rel_model_path)

    loaded_model = UNet_small(in_channels=3, out_channels=2)
    loaded_model.load_state_dict(torch.load(model_path))
    #loaded_model = torch.load(model_path)
    loaded_model.eval()

    embedding = loaded_model(image).squeeze(0).detach().numpy()
    print('Forward Pass Done')

    # bng = get_bandwidth(embedding)
    # print('Bandwidth Estimation Done')
    # print(bng)

    print('Beginning Clustering')
    # result = np.array(cluster_ms(embedding, bandwidth=bng) - 1, np.int)  # labels start at 0


    #result = cluster_agglo(embedding)
    # result = cluster_dbscan(embedding, n_min, epsilon)
    result = cluster_hdbscan(embedding, n_min, epsilon, metric='l2')

    SBD = get_SBD(result, mask.detach().numpy())
    return SBD

    print('Number of Instances Detected:', np.unique(result))
    print('Number of Instances in Ground Truth:', np.unique(mask_example))
    # print('estimates bandwidth:', bng)
    print('Clustering Done')

    if plot is not None:
        fig = plt.figure(figsize=(16, 12))
        plt.title(r'HDBSCAN with $n_m = {}$ and $\epsilon = {}$'.format(n_min, epsilon))
        plt.subplot(1, 3, 1)
        plt.title('Image', size='large')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(np.array(img_example.permute(1, 2, 0)))

        plt.subplot(1, 3, 2)
        plt.xticks([])
        plt.yticks([])
        plt.title('Mask', size='large')
        plt.imshow(mask_example.permute(1, 2, 0), cmap='Spectral', interpolation='nearest')

        plt.subplot(1, 3, 3)
        plt.title('Predicted Mask', size='large')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(result, cmap='Spectral', interpolation='nearest')

        if save is not None:
            fig.savefig('Segmentation.png', dpi=200)
        plt.show()

    #mask_example = np.array(mask_example.detach().numpy(), int)


def apply_on_val_set(n_min, epsilon, method = 'hdbscan', save_images = None):

    assert method in ['hdbscan', 'dbscan', 'meanshift']

    if method == 'hdbscan':
        clustering = cluster_hdbscan
    elif method == 'dbscan':
        clustering = cluster_dbscan
    elif method == 'meanshift':
        clustering = cluster_ms

    HEIGHT, WIDTH = 512, 512

    rel_model_path = '~/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/video/video_small/epoch-900-dim2-s512.pt'
    model_path = os.path.expanduser(rel_model_path)

    loaded_model = UNet_small(in_channels=3, out_channels=2)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()


    rel_path = '~/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'
    directory = os.path.expanduser(rel_path)

    output_directory = os.path.expanduser('~/Documents/BA_Thesis/Data_Dump/Prediction_DB_Val_512/')

    Plants = CustomDatasetMultiple(dir=directory,
                                   transform=None,
                                   image_transform=image_train_transform(HEIGHT, WIDTH),
                                   mask_transform=mask_train_transform(HEIGHT, WIDTH))

    torch.manual_seed(0)
    random.seed(0)
    train_set, val_set, test_set = torch.utils.data.random_split(Plants, [80, 28, 20])

    val_set_split1, val_set_split2 = torch.utils.data.random_split(val_set, [8, 20])

    val_set=val_set_split1
    print(type(train_set))

    running_counting_score = []
    running_SBD = []

    print('Entering Clustering Mode')

    for i in tqdm(range(0,len(val_set),4)):

        item, mask = val_set.__getitem__(i)

        item = item.unsqueeze(0)
        mask = mask.squeeze()
        embedding = loaded_model(item).squeeze(0).detach().numpy()

        pred = clustering(embedding, n_min, epsilon)

        if save_images is not None:
            plt.imshow(pred, cmap='Spectral', interpolation='nearest')
            plt.savefig(output_directory+'pred_image{}.png'.format(i), dpi=200)

            plt.imshow(item.squeeze(0).permute(1,2,0), interpolation='nearest')
            plt.savefig(output_directory+'image{}.png'.format(i), dpi=200)

            plt.imshow(mask, interpolation = 'nearest')
            plt.savefig(output_directory + 'mask{}.png'.format(i), dpi=200)


        SBD = get_SBD(pred, mask.detach().numpy())
        DiC = counting_score(pred, mask)

        running_SBD = np.append(running_SBD, SBD)
        running_counting_score = np.append(running_counting_score, DiC)

#enter information on clustering and size
    with open(output_directory + 'summary-{}-{}-{}.csv'.format(HEIGHT, method, n_min), 'w') as f:
        writer = csv.writer(f)

        header = ['#', 'SBD', '|DiC|']
        writer.writerow(header)

        for j, c in enumerate(running_SBD):
            writer.writerow([str(j), str(running_SBD[j]), str(running_counting_score[j])])

        writer.writerow(['mean', running_SBD.mean(), running_counting_score.mean()])

    #print('The scores for {} for this validation set are:'.format(method))
    #print('Mean Counting Score:', running_counting_score.mean())
    #print('Mean Symmetric Best Dice:', running_SBD.mean())
    #print('Process Finished')

    return running_counting_score.mean(), running_SBD.mean()


if __name__ == '__main__':

    cluster_single_image()
    #score1 = []
    #score2 = []

    #print('Start Searching for best n_min with fixed epsilon')
    #for n in [50, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250, 300, 350, 400]:
     #   SBD = cluster_single_image(n_min=n, epsilon=0.5)
        #score1.append(DiC)
      #  score2.append(SBD)

    #plt.plot(range(0,len(score1)), score1, label='DiC')
    #plt.plot(range(0,len(score2)), score2, label='SBD')
    #plt.legend(borderpad = True)

    #plt.savefig('Optimization_of n_min.png', dpi=200)
    #plt.show()
    

    #cluster_single_image()


    #apply_on_val_set(n_min = 250, epsilon=0.5, method = 'hdbscan')
    #cluster_single_image(save=None, index = 13, n_min=250, epsilon=0.5)



