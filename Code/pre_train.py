import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import matplotlib.pyplot as plt

from model_from_spoco import UNet_spoco
from Custom_Loss.pp_loss import DiscriminativeLoss

from Preprocessing.dataset_plants_binary import CustomDatasetBinary
from Preprocessing.plant_transforms import image_train_transform, mask_train_transform


def pre_trainer():
    """
    The purpose of pretraining is to separete the background pixels from the instance
    pixels by a broad margin.
    This later ameliorates clustering of the foreground as the background pixels
    are not interspersed but clearly separated from the leave clusters.
    """
    LEARNING_RATE = 0.001 #1e-3 empfohlen
    #lambda_1 = lambda epoch: 0.5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 2
    HEIGHT, WIDTH = 300, 300
    IN_CHANNELS = 3  # RGB
    OUT_CHANNELS = 16 #output dimensions of embedding space

    DELTA_VAR = 0.5
    DELTA_D = 2.0

    torch.manual_seed(2)
    #torch.manual_seed(0)

    image_directory = '/Users/luisa/Documents/BA_Thesis/Datasets for Multiple Instance Seg/CVPPP2017_instances/training/A1'

    Plants = CustomDatasetBinary(image_directory,
                                   transform=None,
                                   image_transform= image_train_transform(HEIGHT, WIDTH),
                                   mask_transform= mask_train_transform(HEIGHT, WIDTH)
                                   )

    train_set, val_set = torch.utils.data.random_split(Plants, [100, 28])

    dataloader = DataLoader(train_set, batch_size=8, shuffle=True)
    validation_loader = DataLoader(val_set, batch_size=8, shuffle = True)


    #model_dir = '/Users/luisa/Documents/BA_Thesis/Embedding UNet/Code/saved_models/multi_seg'
    model_dir = '/Users/luisa/Documents/BA_Thesis/Embedding UNet/Code/saved_models/pretraining'
    model = UNet_spoco(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    #scheduler = lr_scheduler.MultiplicativeLR(optimizer, lambda_1)

    loss_function = DiscriminativeLoss(delta_var=DELTA_VAR, delta_d=DELTA_D)
    writer = SummaryWriter('runs/pretrain_runs')

    loss_statistic = np.array([])
    validation_loss_statistic = np.array([])

    for i in range(0, EPOCHS):
        print('Entering Training Epoch {} out of {}'.format(i, EPOCHS))

        #model.train()
        running_loss = 0

        for images, targets in dataloader:
            optimizer.zero_grad()
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            #Predict
            preds = model(images)
            loss = loss_function(preds, targets)

            #Train
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(loss.item())

            #torch.save(model, os.path.join(model_dir, 'epoch-{}.pt'.format(i)))


        #scheduler.step()
        loss_statistic = np.append(loss_statistic, running_loss)

        #Validation Loss
        with torch.no_grad():
            running_validation_loss = 0
            for images, targets in validation_loader:
                predictions = model(images)
                validation_loss = loss_function(predictions, targets)
                running_validation_loss += validation_loss

        validation_loss_statistic = np.append(validation_loss_statistic, running_validation_loss)

        #Tensorboard
        writer.add_scalar('Loss/training_loss of batch', running_loss, i)
        writer.flush()

       # loss_output_epoch = train_function(dataloader, model, optimizer, loss_function, DEVICE)

        #torch.save(model, os.path.join(model_dir, 'epoch-{}.pt'.format(i)))

        print('')
        print('Completed {}/{} Epochs of Training'.format(i + 1, EPOCHS))

    torch.save(model, os.path.join(model_dir, 'pretrain_epoch-{}.pt'.format(EPOCHS)))

    plt.scatter(np.linspace(1, EPOCHS, EPOCHS), loss_statistic/100, label = 'Train Loss')
    plt.scatter(np.linspace(1, EPOCHS, EPOCHS), validation_loss_statistic/28, label = 'Validation Loss')
    plt.title('Train and Validation Loss Per Epoch per Image-Mask-Sample')

    plt.grid()
    plt.xlabel('Training Epoch')
    plt.ylabel('Training Loss')
    plt.yscale('log')
    plt.legend(borderpad = True )
    plt.show()

if __name__ == '__main__':
    pre_trainer()
