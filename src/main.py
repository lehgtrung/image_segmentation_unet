
from utils import *

from dataset import DRIVEDataset
from losses import ContourLoss, ContourLossV3, ContourLossV2, ContourLossV4, BinaryCrossEntropyLoss2d, DiceLoss, FocalLoss
from unet import UNet1024
from fcn32 import FCN8s
import argparse
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

# ------------Path of the images --------------------------------------------------------------
# train
DRIVE_train_imgs_original = "../DRIVE_datasets_training_testing/DRIVE_dataset_imgs_train.hdf5"
DRIVE_train_groudTruth = "../DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_train.hdf5"
DRIVE_train_narrowBand = "../DRIVE_datasets_training_testing/DRIVE_dataset_narrowBand_train.hdf5"

DRIVE_test_imgs_original = "../DRIVE_datasets_training_testing/DRIVE_dataset_imgs_test.hdf5"
DRIVE_test_groundTruth = "../DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_test.hdf5"
DRIVE_test_narrowBand = "../DRIVE_datasets_training_testing/DRIVE_dataset_narrowBand_test.hdf5"
DRIVE_test_border_masks = "../DRIVE_datasets_training_testing/DRIVE_dataset_borderMasks_test.hdf5"

patch_height = 48
patch_width = 48
N_subimgs = 190000
# N_subimgs = 200
inside_FOV = False
test_border_masks = load_hdf5(DRIVE_test_border_masks)
# ---------------------------------------------------------------------------------------------


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='Train or test', dest='mode', default='train', required=True)
    parser.add_argument('--modelpath', help='Path to model for testing', dest='modelpath', required=False)
    parser.add_argument('--epochs', help='Number of epochs', dest='epochs', type=int, default=1000, required=False)
    parser.add_argument('--opt', help='Optimizer', dest='opt', default='adam', required=False)
    parser.add_argument('--hashcode', help='Hashcode for experiments', dest='hashcode', default='', required=False)
    parser.add_argument('--lr', help='Learning rate', dest='lr', default=1e-5, type=float, required=False)
    parser.add_argument('--lossf', help='Loss type', dest='lossf', default='bce')
    parser.add_argument('--gpu', help='Which gpu', dest='gpu', required=False, default='0')
    parser.add_argument('--withlen', help='With contour len', dest='withlen', default='false', required=False)
    parser.add_argument('--mu', help='Value of mu', dest='mu', required=False, default=1, type=float)
    parser.add_argument('--alpha', help='Value of alpha', dest='alpha', required=False, default=1, type=float)
    parser.add_argument('--beta', help='Value of beta', dest='beta', required=False, default=1, type=float)
    parser.add_argument('--normed', help='Normalized contour', dest='normed', required=False, default='true')
    parser.add_argument('--batchsize', help='Batch size', dest='batchsize', required=False, default=16, type=int)
    parser.add_argument('--model', help='Model type FCN or Unet', dest='model', required=False, default='fcn')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    mode = args.mode
    lossf = args.lossf.lower()
    n_epochs = args.epochs
    opt = args.opt.lower()
    lr = args.lr
    hash_code = '_'.join(list(map(str, [args.hashcode, args.opt, args.lr, args.lossf])))
    device = 'cuda:' + args.gpu
    withlen = args.withlen.lower() == 'true'
    normed = args.normed.lower() == 'true'
    mu = args.mu
    alpha = args.alpha
    beta = args.beta
    modelpath = args.modelpath
    DEFAULT_BATCHSIZE = args.batchsize
    modeltype = args.model
    if mode == 'test' and not modelpath:
        raise ValueError("Need model path for testing!!")

    if mode == 'train':
        DRIVE_train = DRIVEDataset(
            "train",
            DRIVE_train_imgs_original,
            DRIVE_train_groudTruth,
            DRIVE_train_narrowBand,
            patch_height,
            patch_width,
            N_subimgs,
            val_size=0
        )
        print(len(DRIVE_train))

        # DRIVE_valid = DRIVE_train.get_validation_dataset()
        DRIVE_train_load = \
            torch.utils.data.DataLoader(dataset=DRIVE_train,
                                        batch_size=DEFAULT_BATCHSIZE, shuffle=True)

        # DRIVE_val_load = \
        #     torch.utils.data.DataLoader(dataset=DRIVE_valid,
        #                                 batch_size=128, shuffle=True)
        DRIVE_valid = DRIVEDataset(
            "test",
            DRIVE_test_imgs_original,
            DRIVE_test_groundTruth,
            DRIVE_test_narrowBand,
            patch_height,
            patch_width,
            None
        )
        DRIVE_val_load = \
            torch.utils.data.DataLoader(dataset=DRIVE_valid,
                                        batch_size=DEFAULT_BATCHSIZE, shuffle=False)

    else:
        DRIVE_test = DRIVEDataset(
            "test",
            DRIVE_test_imgs_original,
            DRIVE_test_groundTruth,
            DRIVE_test_narrowBand,
            patch_height,
            patch_width,
            None
        )
        DRIVE_test_load = \
            torch.utils.data.DataLoader(dataset=DRIVE_test,
                                        batch_size=DEFAULT_BATCHSIZE, shuffle=False)

    shape = (1, 48, 48)
    if torch.cuda.is_available():
        if modeltype.lower() == 'fcn':
            model = FCN8s(n_class=1).to(device)
        else:
            model = UNet1024(shape).to(device)
    else:
        if modeltype.lower() == 'fcn':
            model = FCN8s(n_class=1)
        else:
            model = UNet1024(shape)

    if lossf == 'bce':
        criterion = BinaryCrossEntropyLoss2d()
    elif lossf == 'dice':
        criterion = DiceLoss()
    elif lossf == 'contour':
        criterion = ContourLoss(device=device, mu=mu, alpha=alpha, beta=beta, normed=normed, withlen=withlen)
    elif lossf == 'contour-v3':
        criterion = ContourLossV3(device=device, mu=mu, alpha=alpha, beta=beta, normed=normed, withlen=withlen)
    elif lossf == 'contour-v2':
        criterion = ContourLossV2(device=device, mu=mu, alpha=alpha, beta=beta, normed=normed, withlen=withlen)
    elif lossf == 'contour-v4':
        criterion = ContourLossV4(device=device, mu=mu, normed=normed, withlen=withlen)
    elif lossf == 'focal':
        criterion = FocalLoss()
    else:
        raise ValueError('Undefined loss type')

    optimizer = None
    if opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    if opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Saving History to csv
    header = ['epoch', 'train loss', 'train auc', 'train accuracy',
              'val loss', 'val auc', 'val accuracy', 'val jaccard',
              'val sensitivity', 'val specitivity', 'val precision',
              'val f1', 'val pr auc']
    save_file_name = f"../history/{hash_code}/history.csv"
    save_dir = f"../history/{hash_code}"

    # Saving images and models directories
    model_save_dir = f"../history/{hash_code}/saved_models"
    image_save_path = f"../history/{hash_code}/result_images"

    # Train
    if mode == 'train':
        print("Initializing Training!")
        min_loss = 1e9
        best_epoch = 0
        early_stop_count = 0
        max_count = 5
        for i in range(n_epochs):
            train_loss, train_acc = train_model(i, model, DRIVE_train_load,
                                                criterion, optimizer, device)
            print('Epoch', str(i+1),
                  'Train loss:', train_loss,
                  'Train acc:', train_acc
                  )

            if (i+1) % 5 == 0:
                # val_loss, val_acc = validate_model(model, DRIVE_val_load, criterion, device)
                # print('Epoch', str(i+1),
                #       'Val loss:', val_loss,
                #       'Val acc:', val_acc,
                #       )

                val_loss, val_auc, val_accuracy, val_jaccard, val_sensitivity,\
                    val_specitivity, val_precision, val_f1, val_pr_auc, val_iou = test_model(model, DRIVE_val_load, criterion,
                                                                                             test_border_masks,
                                                                                             f'{image_save_path}/{i+1}/', device)

                print('Epoch', str(i+1),
                      'Val loss:', val_loss,
                      'Val acc:', val_accuracy,
                      )

                values = [i + 1, train_loss, train_acc,
                          val_loss, val_auc, val_accuracy, val_jaccard, val_sensitivity,
                          val_specitivity, val_precision, val_f1, val_pr_auc, val_iou]
                export_history(header, values, save_dir, save_file_name)

                if val_loss < min_loss:
                    early_stop_count = 0
                    min_loss = val_loss
                    best_epoch = i
                    save_model(model, model_save_dir, i+1)
                else:
                    early_stop_count += 1
                    if early_stop_count > max_count:
                        print('Traning can not improve from epoch {}\tBest loss: {}'.format(best_epoch, min_loss))
                        break
    else:
        print("Initializing Testing!")
        model.load_state_dict(torch.load(modelpath))
        model.eval()
        test_auc, test_accuracy, test_jaccard, test_sensitivity, test_specitivity, \
            test_precision, test_f1, test_pr_auc \
            = test_model_img(model,
                             DRIVE_test_load,
                             test_border_masks,
                             f'{image_save_path}/',
                             device)
        print('Test auc: ', test_auc)
        print('Test accuracy: ', test_accuracy)
        print('Test jaccard: ', test_jaccard)
        print('Test sensitivity: ', test_sensitivity)
        print('Test specitivity: ', test_specitivity)
        print('Test precision: ', test_precision)
        print('Test f1: ', test_f1)
        print('Test PR-AUC: ', test_pr_auc)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print('AUC mean: {} - std: {}'.format(*mean_std(test_auc)))
        print('ACCURACY mean: {} - std: {}'.format(*mean_std(test_accuracy)))
        print('JACCARD mean: {} - std: {}'.format(*mean_std(test_jaccard)))
        print('SENS mean: {} - std: {}'.format(*mean_std(test_sensitivity)))
        print('SPEC mean: {} - std: {}'.format(*mean_std(test_specitivity)))
        print('PRECISION mean: {} - std: {}'.format(*mean_std(test_precision)))
        print('F1 mean: {} - std: {}'.format(*mean_std(test_f1)))
        print('PR AUC: {} - std: {}'.format(*mean_std(test_pr_auc)))
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")


if __name__ == "__main__":
    main()

