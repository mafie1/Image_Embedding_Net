# binary Segmentation
# image_directory = '/Users/luisa/Documents/BA_Thesis/CVPPP2015_LCC_training_data/A1'
# Plants = CustomDataset(image_directory, transform=train_transform(HEIGHT, WIDTH))
# model_dir = '/Users/luisa/Documents/BA_Thesis/Embedding UNet/Code/saved_models/binary_seg'
# writer = SummaryWriter('runs/binary_runs')



#Plants = CustomDatasetMultiple(image_directory, transform=train_transform(HEIGHT, WIDTH))

# loss_output_epoch, variance_loss, distance_loss, regularization_loss = train_function(dataloader, model,
#   optimizer, loss_function,
# DEVICE)


def train_function(data, model, optimizer, loss_fn, device):
    print('Entering into train function')

    data = tqdm(data, desc='Training of Batches')
    print('')

    for index, batch in enumerate(data):
        X, y = batch
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        #loss, variance_loss, distance_loss, regularization_loss = loss_fn(preds, y)

        print('Total loss of:', loss.item())

    return loss.item()#, variance_loss.item(), distance_loss.item(), regularization_loss.item()