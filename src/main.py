
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import DRIVEDataset
from losses import ContourLoss, BinaryCrossEntropyLoss2d, DiceLoss
from unet import UNet1024
from preprocess.extract_patches import recompone, kill_border
from preprocess.help_functions import visualize, group_images
from PIL import Image
import argparse
import numpy as np
import os
import csv


# ------------Path of the images --------------------------------------------------------------
# train
DRIVE_train_imgs_original = "../DRIVE_datasets_training_testing/DRIVE_dataset_imgs_train.hdf5"
DRIVE_train_groudTruth = "../DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_train.hdf5"

DRIVE_test_imgs_original = "../DRIVE_datasets_training_testing/DRIVE_dataset_imgs_test.hdf5"
DRIVE_test_groundTruth = "../DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_test.hdf5"

patch_height = 48
patch_width = 48
N_subimgs = 190000
# N_subimgs = 200
inside_FOV = False
# ---------------------------------------------------------------------------------------------


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='Number of epochs', dest='epochs', type=int, default=150, required=False)
    parser.add_argument('--opt', help='Optimizer', dest='opt', default='adam', required=False)
    parser.add_argument('--hashcode', help='Hashcode for experiments', dest='hashcode', default='', required=False)
    parser.add_argument('--lr', help='Learning rate', dest='lr', default=1e-5, type=float, required=False)
    parser.add_argument('--lossf', help='Loss type', dest='lossf')
    parser.add_argument('--gpu', help='Which gpu', dest='gpu', required=True)
    args = parser.parse_args()
    return args


def export_history(header, value, folder, file_name):
    """ export data to csv format
    Args:
        header (list): headers of the column
        value (list): values of correspoding column
        folder (list): folder path
        file_name: file name with path
    """
    # if folder does not exists make folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_existence = os.path.isfile(file_name)

    # if there is no file make file
    if not file_existence:
        file = open(file_name, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(value)
    # if there is file overwrite
    else:
        file = open(file_name, 'a', newline='')
        writer = csv.writer(file)
        writer.writerow(value)
    # close file when it is done with writing
    file.close()


def save_models(model, path, epoch):
    """Save model to given path
    Args:
        model: model to be saved
        path: path that the model would be saved
        epoch: the epoch the model finished training
    """
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model, path+"/model_epoch_{0}.pwf".format(epoch))


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


def train_model(epoch, model, data_train, criterion, optimizer, device):
    model.train()
    losses = []
    accuracies = []
    for batch, (images, masks) in enumerate(data_train):
        if torch.cuda.is_available():
            images = images.to(device)
            masks = masks.to(device)
        images = Variable(images)
        masks = Variable(masks)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = metric_calculator((outputs > 0.0).float(), masks)
        losses.append(loss.item())
        accuracies.append(accuracy)
        if batch % 100 == 0:
            print('Epoch', str(epoch + 1), 'Batch', str(batch + 1), 'Train loss:', loss.item(), "Train acc", accuracy)

    return sum(losses)/len(losses), sum(accuracies)/len(accuracies)


def validate_model(model, data_test, criterion, dirname, device):
    model.eval()
    losses = []
    accuracies = []
    all_outputs = []
    all_images = []
    all_masks = []
    os.makedirs(dirname, exist_ok=True)
    for batch, (images, masks) in enumerate(data_test):
        with torch.no_grad():
            if torch.cuda.is_available():
                images = images.to(device)
                masks = masks.to(device)
            images = Variable(images)
            masks = Variable(masks)
            outputs = model(images)
            loss = criterion(outputs, masks)

            thresholded_outputs = (outputs > 0.0).float()
            all_outputs.append(thresholded_outputs.detach().cpu())
            all_images.append(images.detach().cpu())
            all_masks.append(masks.detach().cpu())

            accuracy = metric_calculator(thresholded_outputs, masks)
            losses.append(loss.item())
            accuracies.append(accuracy)
            if batch % 1 == 0:
                print('Batch', str(batch + 1), 'Val loss:', loss.item(), "Val acc", accuracy)

    # TODO: output predicted images
    # print(all_outputs)
    all_outputs = torch.cat(all_outputs)
    all_images = torch.cat(all_images)
    all_masks = torch.cat(all_masks)
    pred_imgs = recompone(all_outputs, 13, 12)  # predictions
    orig_imgs = recompone(all_images, 13, 12)  # originals
    gtruth_masks = recompone(all_masks, 13, 12)  # masks
    # kill_border(pred_imgs, test_border_masks)
    # back to original dimensions
    orig_imgs = orig_imgs[:, :, 0:565, 0:584]
    pred_imgs = pred_imgs[:, :, 0:565, 0:584]
    gtruth_masks = gtruth_masks[:, :, 0:565, 0:584]
    visualize(group_images(orig_imgs, 1), dirname + "all_originals")
    visualize(group_images(pred_imgs, 1), dirname + "all_predictions")
    visualize(group_images(gtruth_masks, 1), dirname + "all_masks")

    return sum(losses)/len(losses), sum(accuracies)/len(accuracies)


def main():
    DRIVE_train = DRIVEDataset(
        "train",
        DRIVE_train_imgs_original,
        DRIVE_train_groudTruth,
        patch_height,
        patch_width,
        N_subimgs
    )

    DRIVE_test = DRIVEDataset(
        "test",
        DRIVE_test_imgs_original,
        DRIVE_test_groundTruth,
        patch_height,
        patch_width,
        -1
    )

    SEM_train_load = \
        torch.utils.data.DataLoader(dataset=DRIVE_train,
                                    batch_size=32, shuffle=True)
    SEM_test_load = \
        torch.utils.data.DataLoader(dataset=DRIVE_test,
                                    batch_size=32, shuffle=False)

    args = arg_parse()
    lossf = args.lossf.lower()
    n_epochs = args.epochs
    opt = args.opt.lower()
    lr = args.lr
    hash_code = '_'.join(list(map(str, [args.hashcode, args.opt, args.lr, args.lossf])))
    device = 'cuda:' + args.gpu

    shape = (1, 48, 48)
    if torch.cuda.is_available():
        model = UNet1024(shape).to(device)
    else:
        model = UNet1024(shape)

    if lossf == 'bce':
        criterion = BinaryCrossEntropyLoss2d()
    elif lossf == 'dice':
        criterion = DiceLoss()
    elif lossf == 'contour':
        criterion = ContourLoss()
    else:
        raise ValueError('Undefined loss type')

    optimizer = None
    if opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    if opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Saving History to csv
    header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
    save_file_name = f"../history/{hash_code}/history.csv"
    save_dir = f"../history/{hash_code}"

    # Saving images and models directories
    model_save_dir = f"../history/{hash_code}/saved_models"
    image_save_path = f"../history/{hash_code}/result_images"

    # Train
    print("Initializing Training!")
    for i in range(n_epochs):
        # train the model
        train_loss, train_acc = train_model(i, model, SEM_train_load, criterion, optimizer, device)
        print('Epoch', str(i+1), 'Train loss:', train_loss, "Train acc", train_acc)

        # Validation every 10 epoch
        if (i+1) % 1 == 0:
            val_loss, val_acc = validate_model(model, SEM_test_load, criterion,
                                               f'{image_save_path}/{i+1}/', device)
            print('Epoch', str(i+1), 'Val loss:', val_loss, "val acc:", val_acc)
            values = [i + 1, train_loss, train_acc, val_loss, val_acc]
            export_history(header, values, save_dir, save_file_name)

        if (i+1) % 3 == 0:  # save model every 10 epoch
            save_models(model, model_save_dir, i+1)


if __name__ == "__main__":
    main()

