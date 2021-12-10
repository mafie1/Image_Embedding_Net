import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import random
from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple, image_train_transform, mask_train_transform
from Custom_Loss.pp_loss import DiscriminativeLoss
from model import UNet_spoco, UNet_small, UNet_spoco_new
from tqdm import tqdm
from utils import plot_results_from_training
import shutil
from params import *

def trainer():

    #os.path.abspath(os.path.join(file, '../../helper_data/fmh_title_numeric_mapping.json')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Working on: ', DEVICE)

    print('Training on images of size {}x{}'.format(HEIGHT, WIDTH), '\n')

    re_img_dir = '~/Documents/BA_Thesis/CVPPP2017_instances/training/A1'
    image_directory = os.path.expanduser(re_img_dir)
    print('Image Directory is set to:', image_directory, '\n')

    Plants = CustomDatasetMultiple(image_directory,
                                   transform=None,
                                   image_transform=image_train_transform(HEIGHT, WIDTH),
                                   mask_transform= mask_train_transform(HEIGHT, WIDTH))

    torch.manual_seed(0)
    random.seed(0)
    train_set, val_set, test_set = torch.utils.data.random_split(Plants, [80, 28, 20])

    dataloader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    validation_loader = DataLoader(val_set, batch_size=8, shuffle = False, num_workers=4)


    """Choose right Model and create Directory for Run with Run Parameters"""

    if big is not None:
        model = UNet_spoco_new(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
        model.train()

        rel_model_dir = '~/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/full_UNet/'
        model_dir = os.path.expanduser(rel_model_dir)

        created_model_dir = model_dir + 'run-dim{}-height{}-epochs{}'.format(OUT_CHANNELS, HEIGHT, EPOCHS)

        if (os.path.exists(created_model_dir)):
            print('Directory already exists and will be overwritten \n')
            shutil.rmtree(created_model_dir)

        os.makedirs(created_model_dir)
        model_dir = created_model_dir
        print('The Model will be saved in:', model_dir)

    else:
        model = UNet_small(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
        model.train()

        rel_model_dir = '~/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/small_UNet/'
        model_dir = os.path.expanduser(rel_model_dir)

        created_model_dir = model_dir + 'run-dim{}-height{}-epochs{}'.format(OUT_CHANNELS, HEIGHT, EPOCHS)

        if (os.path.exists(created_model_dir)):
            print('Directory already exists and will be overwritten \n')
            shutil.rmtree(created_model_dir)

        os.makedirs(created_model_dir)
        model_dir = created_model_dir
        print('The Model will be saved in:', model_dir)


    """Counting Model Parameters"""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('# Model Parameters: ', params, '\n')


    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_function = DiscriminativeLoss(delta_var=DELTA_VAR, delta_d=DELTA_D)

    loss_statistic = np.array([])
    validation_loss_statistic = np.array([])

    for i in tqdm(range(0, EPOCHS)):
        model.train()
        running_loss = 0

        for images, targets in dataloader:
            optimizer.zero_grad()
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            #Predict
            preds = model(images).to(DEVICE)
            loss = loss_function(preds, targets)

            #Train
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #print(loss.item())

        if i in [0, 2, 5, 10, 20, 30, 40, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1500, 2000]:
            torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}-dim{}-s{}.pt'.format(i, OUT_CHANNELS, HEIGHT)))

        #scheduler.step()
        loss_statistic = np.append(loss_statistic, running_loss)

        """Validation Loss"""
        model.eval()
        with torch.no_grad():
            running_validation_loss = 0
            for images, targets in validation_loader:
                predictions = model(images)
                validation_loss = loss_function(predictions, targets)
                running_validation_loss += validation_loss

        validation_loss_statistic = np.append(validation_loss_statistic, running_validation_loss)

    print('Training Done')
    np.savetxt(model_dir+"/training_loss.csv", [np.linspace(1, EPOCHS, EPOCHS) ,loss_statistic/len(test_set), validation_loss_statistic/len(val_set)], delimiter=",")

    """Save Model after Training"""
    torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}-dim{}-s{}.pt'.format(EPOCHS, OUT_CHANNELS, HEIGHT)))

    """Make plot for Loss Statistics"""

    plot_results_from_training(EPOCHS, losses=loss_statistic, val_losses=validation_loss_statistic, save_path= model_dir)


if __name__ == '__main__':
    trainer()
