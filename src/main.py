
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import DRIVEDataset
from losses import ContourLoss, BinaryCrossEntropyLoss2d, DiceLoss
from unet import UNet1024
from original_unet import OriginalUnet
from PIL import Image
import argparse
import numpy as np


# ------------Path of the images --------------------------------------------------------------
# train
DRIVE_train_imgs_original = "../DRIVE_datasets_training_testing/DRIVE_dataset_imgs_train.hdf5"
DRIVE_train_groudTruth = "../DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_train.hdf5"

DRIVE_test_imgs_original = "../DRIVE_datasets_training_testing/DRIVE_dataset_imgs_test.hdf5"
DRIVE_test_groundTruth = "../DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_test.hdf5"

patch_height = 48
patch_width = 48
N_subimgs = 190000
inside_FOV = True
# ---------------------------------------------------------------------------------------------


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='Number of epochs', dest='epochs', type=int, default=500, required=False)
    parser.add_argument('--opt', help='Optimizer', dest='opt', default='adam', required=False)
    parser.add_argument('--hashcode', help='Hashcode for experiments', dest='hashcode', default='NONE', required=False)
    parser.add_argument('--lr', help='Learning rate', dest='lr', default=1e-5, type=float, required=False)
    parser.add_argument('--lossf', help='Learning rate', dest='lossf')
    args = parser.parse_args()
    return args


def metric_calculator(predictions, masks):
    def accuracy_check(prediction, mask):
        ims = [mask, prediction]
        np_ims = []
        for item in ims:
            if 'str' in str(type(item)):
                item = np.array(Image.open(item))
            elif 'PIL' in str(type(item)):
                item = np.array(item)
            elif 'torch' in str(type(item)):
                if torch.cuda.is_available():
                    item = item.cpu().numpy()
                else:
                    item = item.numpy()
            np_ims.append(item)

        compare = np.equal(np_ims[0], np_ims[1])
        accuracy = np.sum(compare)

        return accuracy / len(np_ims[0].flatten())

    batch_size = predictions.size(0)
    total_acc = 0
    for index in range(batch_size):
        total_acc += accuracy_check(predictions[index], masks[index])
    return total_acc / batch_size


def train_model(model, data_train, criterion, optimizer):
    model.train()
    for batch, (images, masks) in enumerate(data_train):
        if torch.cuda.is_available():
            images = images.cuda()
            masks = masks.cuda()
        images = Variable(images)
        masks = Variable(masks)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), metric_calculator((outputs > 0.0).float(), masks)


def test_model(model, data_test, criterion):
    model.train()
    for batch, (images, masks) in enumerate(data_test):
        with torch.no_grad():
            if torch.cuda.is_available():
                images = images.cuda()
                masks = masks.cuda()
            images = Variable(images)
            masks = Variable(masks)
            outputs = model(images)
            loss = criterion(outputs, masks)
            return loss.item(), metric_calculator((outputs > 0.0).float(), masks)


def main():
    DRIVE_train = DRIVEDataset(
        "train",
        DRIVE_train_imgs_original,
        DRIVE_train_groudTruth,
        48,
        48,
        190000
    )

    DRIVE_test = DRIVEDataset(
        "test",
        DRIVE_test_imgs_original,
        DRIVE_test_groundTruth,
        48,
        48,
        -1
    )

    SEM_train_load = \
        torch.utils.data.DataLoader(dataset=DRIVE_train,
                                    num_workers=16, batch_size=2, shuffle=True)
    SEM_test_load = \
        torch.utils.data.DataLoader(dataset=DRIVE_test,
                                    num_workers=3, batch_size=1, shuffle=True)

    args = arg_parse()
    lossf = args.lossf.lower()
    n_epochs = args.epochs
    opt = args.opt.lower()
    lr = args.lr
    hash_code = '_'.join(list(map(str, [args.hashcode, args.opt, args.lr, args.lossf])))

    shape = (1, 48, 48)
    model = UNet1024(shape)
    # model = OriginalUnet(1, 1)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=list(
            range(torch.cuda.device_count()))).cuda()

    if lossf == 'bce':
        criterion = BinaryCrossEntropyLoss2d()
    elif lossf == 'dice':
        criterion = DiceLoss()
    elif lossf == 'contour':
        criterion = ContourLoss()
    else:
        raise ValueError('Undefined loss type')

    optimizer = None
    if torch.cuda.is_available():
        if opt == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.module.parameters(), lr=lr)
        if opt == 'adam':
            optimizer = torch.optim.Adam(model.module.parameters(), lr=lr)
    else:
        if opt == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        if opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Saving History to csv
    # header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
    # save_file_name = f"../history/{hash_code}/history.csv"
    # save_dir = f"../history/{hash_code}"

    # Saving images and models directories
    # model_save_dir = f"../history/{hash_code}/saved_models"
    # image_save_path = f"../history/{hash_code}/result_images"

    # Train
    print("Initializing Training!")
    for i in range(n_epochs):
        # train the model
        train_loss, train_accuracy = train_model(model, SEM_train_load, criterion, optimizer)
        print('Epoch', str(i+1), 'Train loss:', train_loss, "Train acc", train_accuracy)

        # Validation every 5 epoch
        if (i+1) % 10 == 0:
            val_loss, val_accuracy = test_model(
                model, SEM_test_load, criterion)
            print('Val loss:', val_loss, "val acc:", val_accuracy)


if __name__ == "__main__":
    main()

