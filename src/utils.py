import torch
import torch.nn.functional as F
from torch.autograd import Variable
from preprocess.extract_patches import recompone, kill_border, recompone_overlap
from preprocess.help_functions import visualize, group_images, load_hdf5
import metrics as mtr
import os
import glob
import csv
import random
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image


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


def metrics_calculator(masks, preds, mode_average=True, additional=False):
    batch_size, masks, predictions = mtr.standardize_for_metrics(masks, preds)
    auc_score = mtr.roc_auc(batch_size, masks, predictions, mode_average)
    accuracy_score = mtr.accuracy(batch_size, masks, predictions, mode_average)
    if additional:
        jaccard_score = mtr.jaccard(batch_size, masks, predictions, mode_average)
        sens_score, spec_score, prec_score = mtr.confusion_matrix(batch_size, masks, predictions, mode_average)
        return auc_score, accuracy_score, jaccard_score, sens_score, spec_score, prec_score
    return auc_score, accuracy_score


def train_model(epoch, model, data_train, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    epoch_auc = 0.0
    epoch_accuracy = 0.0
    n = 0.0
    for batch, (images, masks, narrowbands) in enumerate(data_train):
        n += 1
        if torch.cuda.is_available():
            images = images.to(device)
            masks = masks.to(device)
            narrowbands = narrowbands.to(device)
        images = Variable(images)
        masks = Variable(masks)
        narrowbands = Variable(narrowbands)
        outputs = model(images)
        try:
            loss = criterion(outputs, masks, narrowbands)
        except ValueError:
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
    epoch_loss = .0
    epoch_auc = .0
    epoch_accuracy = .0
    epoch_jaccard = .0
    n = 0.0
    for batch, (images, masks, narrowbands) in enumerate(data_test):
        n += 1
        with torch.no_grad():
            if torch.cuda.is_available():
                images = images.to(device)
                masks = masks.to(device)
                narrowbands = narrowbands.to(device)
            images = Variable(images)
            masks = Variable(masks)
            narrowbands = Variable(narrowbands)
            outputs = model(images)
            try:
                loss = criterion(outputs, masks, narrowbands)
            except ValueError:
                loss = criterion(outputs, masks)
            auc, accuracy, jaccard = metrics_calculator(masks.clone(), outputs.clone())
            epoch_loss += loss.item()

            epoch_auc += auc
            epoch_accuracy += accuracy
            epoch_jaccard += jaccard
            if batch % 100 == 0:
                print('Batch', str(batch + 1),
                      'Val loss:', loss.item(),
                      'Val auc:', auc,
                      'Val accuracy:', accuracy,
                      'Val jaccard:', jaccard
                      )
    return epoch_loss / n, epoch_auc / n, epoch_accuracy / n, epoch_jaccard / n


def test_model_img(model, data_test, test_border_masks, dirname, device):
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

            all_outputs.append(outputs.detach().cpu())
            all_images.append(images.detach().cpu())
            all_masks.append(masks.detach().cpu())

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

    # Put more metrics here
    # Retrive list of performance metric for each image
    # pred_imgs = (pred_imgs >= 0.5).astype('int')

    # After calculating best cut off, recalculate other metrics
    (_, auc), accuracy, jaccard, sensitivity, specitivity, precision \
        = metrics_calculator(gtruth_masks, pred_imgs, mode_average=False, additional=True)

    visualize(group_images(orig_imgs, 1), dirname + "all_originals")
    visualize(group_images(pred_imgs, 1), dirname + "all_predictions")
    visualize(group_images(gtruth_masks, 1), dirname + "all_masks")

    return auc, accuracy, jaccard, sensitivity, specitivity, precision


def extract_narrow_band(input_dir, output_dir, d1=1, d2=1):
    """
        Applying morphological gradient: the difference between dilation and erosion
        https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.morphology.binary_dilation.html
    """
    img_paths = glob.glob(os.path.join(input_dir, '*.gif'))
    os.makedirs(output_dir, exist_ok=True)
    for path in img_paths:
        idx = os.path.basename(path)[:2]
        im = np.asarray(Image.open(path))
        dilation_mask = ndimage.binary_dilation(im, iterations=d1).astype(im.dtype)
        erosion_mask = ndimage.binary_erosion(im, iterations=d2).astype(im.dtype)
        out_im = Image.fromarray(((dilation_mask - erosion_mask) * 255).astype(np.uint8))
        out_im.save(os.path.join(output_dir, idx + "_narr.gif"))


def split_arrays(size, *args):
    indices = list(range(len(args[0])))
    n = int(len(indices) * size)
    random.shuffle(indices)
    print("Len indices", len(indices))
    print("Len n", n)
    for arr in args:
        assert len(arr) == len(indices)
        yield arr[indices][n:], arr[indices][:n]


if __name__ == '__main__':
    # extract_narrow_band('../DRIVE/training/1st_manual',
    #                     '../DRIVE/training/narrowband')
    # extract_narrow_band('../DRIVE/test/1st_manual',
    #                     '../DRIVE/test/narrowband')
    img_path = '../DRIVE/test/2nd_manual/01_manual2.gif'
    img = np.asarray(Image.open(img_path))

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=1, ncols=6, figsize=(12, 6),
                                                       sharex=True, sharey=True)

    dilated_img = ndimage.binary_dilation(img, iterations=2).astype(img.dtype)
    erosed_img = ndimage.binary_erosion(img, iterations=2).astype(img.dtype)
    morp_img = dilated_img - erosed_img
    narr_img = img * morp_img
    # narr_img2 = (1 - img) * morp_img
    narr_img2 = (1 - img) * morp_img

    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('Original image', fontsize=10)

    ax2.imshow(dilated_img, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Dilated image', fontsize=10)

    ax3.imshow(erosed_img, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('Erosed image', fontsize=10)

    ax4.imshow(morp_img, cmap=plt.cm.gray)
    ax4.axis('off')
    ax4.set_title('Morp image', fontsize=10)

    ax5.imshow(narr_img, cmap=plt.cm.gray)
    ax5.axis('off')
    ax5.set_title('target * (1 - mask)', fontsize=10)

    ax6.imshow(narr_img2, cmap=plt.cm.gray)
    ax6.axis('off')
    ax6.set_title('(1 - target) * (1 - mask)', fontsize=10)

    fig.tight_layout()

    plt.show()
