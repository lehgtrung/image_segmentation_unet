import torch
import torch.nn as nn
from torch.autograd import Variable
from preprocess.extract_patches import recompone, kill_border
from preprocess.help_functions import visualize, group_images, load_hdf5
import metrics as mtr
import os
import csv


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


def save_checkpoint(model, epoch, optimizer, path):
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, path+"/checkpoint_epoch_{0}.pth".format(epoch))


def save_model(model, path, epoch):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), path+"/model_epoch_{0}.pth".format(epoch))


def metrics_calculator(masks, preds):
    batch_size, masks, predictions = mtr.standardize_for_metrics(masks, preds)
    auc_score = mtr.roc_auc(batch_size, masks, predictions)
    accuracy_score = mtr.accuracy(batch_size, masks, predictions)
    # jaccard_score = mtr.jaccard(batch_size, masks, predictions)
    return auc_score, accuracy_score


def train_model(epoch, model, data_train, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    epoch_auc = 0.0
    epoch_accuracy = 0.0
    n = 0.0
    for batch, (images, masks) in enumerate(data_train):
        n += 1
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

        auc, accuracy = metrics_calculator(masks.clone(), outputs.clone())
        epoch_loss += loss.item()

        epoch_auc += auc
        epoch_accuracy += accuracy
        if batch % 100 == 0:
            print('Epoch', str(epoch + 1),
                  'Batch', str(batch + 1),
                  'Train loss:', loss.item(),
                  'Train auc:', auc,
                  'Train acc:', accuracy,
                  )
    return epoch_loss/n, epoch_auc/n, epoch_accuracy/n


def validate_model(model, data_test, criterion, device):
    model.eval()
    epoch_loss = 0.0
    epoch_auc = 0.0
    epoch_accuracy = 0.0
    n = 0.0
    for batch, (images, masks) in enumerate(data_test):
        n += 1
        with torch.no_grad():
            if torch.cuda.is_available():
                images = images.to(device)
                masks = masks.to(device)
            images = Variable(images)
            masks = Variable(masks)
            outputs = model(images)
            loss = criterion(outputs, masks)

            auc, accuracy = metrics_calculator(masks.clone(), outputs.clone())
            epoch_loss += loss.item()

            epoch_auc += auc
            epoch_accuracy += accuracy
            if batch % 1 == 0:
                print('Batch', str(batch + 1),
                      'Val loss:', loss.item(),
                      'Val auc:', auc,
                      'Val accuracy:', accuracy,
                      )
    return epoch_loss / n, epoch_auc / n, epoch_accuracy / n


def test_model(model, data_test, criterion, test_border_masks, dirname, device):
    model.eval()
    epoch_loss = 0.0
    all_outputs = []
    all_images = []
    all_masks = []
    epoch_auc = 0.0
    epoch_accuracy = 0.0
    n = 0.0
    os.makedirs(dirname, exist_ok=True)
    for batch, (images, masks) in enumerate(data_test):
        n += 1
        with torch.no_grad():
            if torch.cuda.is_available():
                images = images.to(device)
                masks = masks.to(device)
            images = Variable(images)
            masks = Variable(masks)
            outputs = model(images)
            loss = criterion(outputs, masks)

            thresholded_outputs = (outputs > 0.5).float()
            all_outputs.append(thresholded_outputs.detach().cpu())
            all_images.append(images.detach().cpu())
            all_masks.append(masks.detach().cpu())

            auc, accuracy = metrics_calculator(masks.clone(), outputs.clone())
            epoch_loss += loss.item()

            epoch_auc += auc
            epoch_accuracy += accuracy
            if batch % 1 == 0:
                print('Batch', str(batch + 1),
                      'Val loss:', loss.item(),
                      'Val auc:', auc,
                      'Val accuracy:', accuracy,
                      )

    all_outputs = torch.cat(all_outputs)
    all_images = torch.cat(all_images)
    all_masks = torch.cat(all_masks)
    pred_imgs = recompone(all_outputs, 13, 12)  # predictions
    orig_imgs = recompone(all_images, 13, 12)  # originals
    gtruth_masks = recompone(all_masks, 13, 12)  # masks
    kill_border(pred_imgs, test_border_masks)
    # back to original dimensions
    orig_imgs = orig_imgs[:, :, 0:565, 0:584]
    pred_imgs = pred_imgs[:, :, 0:565, 0:584]
    gtruth_masks = gtruth_masks[:, :, 0:565, 0:584]
    visualize(group_images(orig_imgs, 1), dirname + "all_originals")
    visualize(group_images(pred_imgs, 1), dirname + "all_predictions")
    visualize(group_images(gtruth_masks, 1), dirname + "all_masks")

    return epoch_loss / n, epoch_auc / n, epoch_accuracy / n


def test_model_img(model, data_test, test_border_masks, dirname, device, threshold=0.5):
    model.eval()
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

            thresholded_outputs = (outputs > threshold).float()
            all_outputs.append(thresholded_outputs.detach().cpu())
            all_images.append(images.detach().cpu())
            all_masks.append(masks.detach().cpu())

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

    # Put more metrics here
    auc, accuracy = metrics_calculator(gtruth_masks, pred_imgs)

    visualize(group_images(orig_imgs, 1), dirname + "all_originals")
    visualize(group_images(pred_imgs, 1), dirname + "all_predictions")
    visualize(group_images(gtruth_masks, 1), dirname + "all_masks")

    return auc, accuracy



