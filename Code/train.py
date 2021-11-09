import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import matplotlib.pyplot as plt

from model_from_spoco import UNet_spoco
from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple, image_train_transform, mask_train_transform
from Custom_Loss.pp_loss import DiscriminativeLoss


def trainer():
    torch.manual_seed(3)

    LEARNING_RATE = 0.001 #1e-3 empfohlen
    #lambda_1 = lambda epoch: 0.5

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 200
    HEIGHT =  256
    WIDTH = HEIGHT
    IN_CHANNELS = 3  # RGB
    OUT_CHANNELS = 16 #output dimensions of embedding space

    DELTA_VAR = 0.5
    DELTA_D = 2.5

    print('Training on images of size {}x{}'.format(HEIGHT, WIDTH))


    re_img_dir = '~/Documents/BA_Thesis/CVPPP2017_instances/training/A1'
    image_directory = os.path.expanduser(re_img_dir)
    print('Image Directory is set to:', image_directory, '\n')


    Plants = CustomDatasetMultiple(image_directory,
                                   transform=None,
                                   image_transform= image_train_transform(HEIGHT, WIDTH),
                                   mask_transform= mask_train_transform(HEIGHT, WIDTH)
                                   )

    train_set, val_set, test_set = torch.utils.data.random_split(Plants, [80, 28, 20])

    dataloader = DataLoader(train_set, batch_size=8, shuffle=True)
    validation_loader = DataLoader(val_set, batch_size=8, shuffle = False)


    rel_model_dir = '~/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/time_evolution'

    model_dir = os.path.expanduser(rel_model_dir)
    print('Model Directory is set to:', model_dir)

    model = UNet_spoco(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    #scheduler = lr_scheduler.MultiplicativeLR(optimizer, lambda_1)

    loss_function = DiscriminativeLoss(delta_var=DELTA_VAR, delta_d=DELTA_D)
    #writer = SummaryWriter('runs/multi_runs')

    loss_statistic = np.array([])
    validation_loss_statistic = np.array([])

    #print(optimizer.state_dict())

    for i in range(0, EPOCHS):
        print('Entering Training Epoch {} out of {}'.format(i, EPOCHS))

        model.train()
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

            #print(loss.item())

        if i in [0, 10, 20, 30, 40, 50, 100, 150, 190]:
            torch.save(model, os.path.join(model_dir, 'epoch-{}.pt'.format(i)))


        #scheduler.step()
        loss_statistic = np.append(loss_statistic, running_loss)

        #Validation Loss
        model.eval()
        with torch.no_grad():
            running_validation_loss = 0
            for images, targets in validation_loader:
                predictions = model(images)
                validation_loss = loss_function(predictions, targets)
                running_validation_loss += validation_loss

        validation_loss_statistic = np.append(validation_loss_statistic, running_validation_loss)

        #Tensorboard
        #writer.add_scalar('Loss/training_loss of batch', running_loss, i)
        #writer.flush()

        #Save Model after each Epoch?
        #torch.save(model, os.path.join(model_dir, 'epoch-{}.pt'.format(i)))

        print('')
        print('Completed {}/{} Epochs of Training'.format(i + 1, EPOCHS))


    print('Training Done')
    np.savetxt("training_loss.csv", [np.linspace(1, EPOCHS, EPOCHS) ,loss_statistic/80, validation_loss_statistic/28], delimiter=",")

    #Save Model after entire training?
    torch.save(model, os.path.join(model_dir, 'epoch-{}.pt'.format(EPOCHS)))

    #Make loss statistic plot
    plt.plot(np.linspace(1, EPOCHS, EPOCHS), loss_statistic/80, label = 'Train Loss')
    plt.plot(np.linspace(1, EPOCHS, EPOCHS), validation_loss_statistic/28, label = 'Validation Loss')
    #plt.plot(np.linspace(1, EPOCHS, EPOCHS), loss_statistic/100, label = 'Train Loss')
    #plt.plot(np.linspace(1, EPOCHS, EPOCHS), validation_loss_statistic/28, label = 'Validation Loss')
    plt.title('Train and Validation Loss Per Epoch per Image-Mask-Sample')

    plt.grid()
    plt.xlabel('Training Epoch')
    plt.ylabel('Training Loss')
    plt.yscale('log')
    plt.legend(borderpad = True )

    #plt.show()
    plt.savefig('Loss_statistic.png')


if __name__ == '__main__':
    trainer()
