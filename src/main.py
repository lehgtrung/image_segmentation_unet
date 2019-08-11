
from utils import *

from dataset import DRIVEDataset
from losses import ContourLoss, BinaryCrossEntropyLoss2d, DiceLoss, FocalLoss
from unet import UNet1024
import argparse

# ------------Path of the images --------------------------------------------------------------
# train
DRIVE_train_imgs_original = "../DRIVE_datasets_training_testing/DRIVE_dataset_imgs_train.hdf5"
DRIVE_train_groudTruth = "../DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_train.hdf5"

DRIVE_test_imgs_original = "../DRIVE_datasets_training_testing/DRIVE_dataset_imgs_test.hdf5"
DRIVE_test_groundTruth = "../DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_test.hdf5"
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
    parser.add_argument('--epochs', help='Number of epochs', dest='epochs', type=int, default=150, required=False)
    parser.add_argument('--opt', help='Optimizer', dest='opt', default='adam', required=False)
    parser.add_argument('--hashcode', help='Hashcode for experiments', dest='hashcode', default='', required=False)
    parser.add_argument('--lr', help='Learning rate', dest='lr', default=1e-5, type=float, required=False)
    parser.add_argument('--lossf', help='Loss type', dest='lossf', default='bce')
    parser.add_argument('--gpu', help='Which gpu', dest='gpu', required=True)
    parser.add_argument('--withlen', help='With contour len', dest='withlen', default='false', required=False)
    parser.add_argument('--mu', help='Value of mu', dest='mu', required=False, default=0.1, type=float)
    parser.add_argument('--normed', help='Normalized contour', dest='normed', required=False, default='true')
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
    modelpath = args.modelpath
    if mode == 'test' and not modelpath:
        raise ValueError("Need model path for testing!!")

    if mode == 'train':
        DRIVE_train = DRIVEDataset(
            "train",
            DRIVE_train_imgs_original,
            DRIVE_train_groudTruth,
            patch_height,
            patch_width,
            N_subimgs
        )

        DRIVE_valid = DRIVE_train.get_validation_dataset()
        DRIVE_train_load = \
            torch.utils.data.DataLoader(dataset=DRIVE_train,
                                        batch_size=32, shuffle=True)

        DRIVE_val_load = \
            torch.utils.data.DataLoader(dataset=DRIVE_valid,
                                        batch_size=32, shuffle=True)
    else:
        DRIVE_test = DRIVEDataset(
            "test",
            DRIVE_test_imgs_original,
            DRIVE_test_groundTruth,
            patch_height,
            patch_width,
            -1
        )
        DRIVE_test_load = \
            torch.utils.data.DataLoader(dataset=DRIVE_test,
                                        batch_size=32, shuffle=False)

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
        criterion = ContourLoss(device=device, mu=mu, normed=normed, withlen=withlen)
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
              'val loss', 'val auc', 'val accuracy']
    save_file_name = f"../history/{hash_code}/history.csv"
    save_dir = f"../history/{hash_code}"

    # Saving images and models directories
    model_save_dir = f"../history/{hash_code}/saved_models"
    image_save_path = f"../history/{hash_code}/result_images"

    # Train
    if mode == 'train':
        print("Initializing Training!")
        for i in range(n_epochs):
            # train the model
            train_loss, train_auc, train_acc = train_model(i, model, DRIVE_train_load,
                                                           criterion, optimizer, device)
            print('Epoch', str(i+1),
                  'Train loss:', train_loss,
                  'Train auc:', train_auc,
                  'Train acc:', train_acc,
                  # 'Train jaccard:', train_jaccard
                  )

            if (i+1) % 1 == 0:
                # val_loss, val_auc, val_acc = validate_model(model, DRIVE_val_load, criterion,
                #                                             f'{image_save_path}/{i+1}/', device)
                val_loss, val_auc, val_acc = validate_model(model, DRIVE_val_load, criterion, device)
                print('Epoch', str(i+1),
                      'Val loss:', val_loss,
                      'Val auc:', val_auc,
                      'Val acc:', val_acc,
                      # 'Val jaccard:', val_jaccard
                      )
                # values = [i + 1, train_loss, train_auc, train_acc, train_jaccard,
                #           val_loss, val_auc, val_acc, val_jaccard]
                values = [i + 1, train_loss, train_auc, train_acc,
                          val_loss, val_auc, val_acc]
                export_history(header, values, save_dir, save_file_name)

            if (i+1) % 10 == 0:  # save model every 1 epoch
                save_model(model, model_save_dir, i+1)
    else:
        print("Initializing Testing!")
        model.load_state_dict(torch.load(modelpath))
        model.eval()
        test_auc, test_accuracy = test_model_img(model,
                                                 DRIVE_test_load,
                                                 test_border_masks,
                                                 f'{image_save_path}/',
                                                 device)
        print('Test auc: ', test_auc)
        print('Test accuracy: ', test_accuracy)


if __name__ == "__main__":
    main()

