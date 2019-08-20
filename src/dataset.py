import torch
import numpy as np
from utils import split_arrays
from torch.utils.data.dataset import Dataset
from preprocess.extract_patches import get_data_training, get_data_testing


class DRIVEDataset(Dataset):

    def __init__(self, mode, img_path, groudtruth_path, narrowband_path, height, width,
                 n_subimgs, inside_fov=True, val_size=0.1):
        if mode == 'train':
            self.img_patches, self.groundtruth_patches, self.narrowband_patches = \
                get_data_training(img_path, groudtruth_path, narrowband_path,
                                  height, width, n_subimgs, inside_fov)
            (self.img_patches, self.val_img_patches), \
            (self.groundtruth_patches, self.val_groundtruth_patches), \
            (self.narrowband_patches, self.val_narrowband_patches) = split_arrays(val_size, self.img_patches,
                                                                                  self.groundtruth_patches,
                                                                                  self.narrowband_patches)
        elif mode == 'test':
            if isinstance(img_path, np.ndarray):
                self.img_patches, self.groundtruth_patches, self.narrowband_patches = img_path, \
                                                                                      groudtruth_path, narrowband_path
            else:
                self.img_patches, self.groundtruth_patches = get_data_testing(img_path, groudtruth_path,
                                                                              n_subimgs, height, width)
        else:
            raise ValueError("mode needs to be train | test")
        self.data_len = len(self.img_patches)

    def get_validation_dataset(self):
        return DRIVEDataset('test', self.val_img_patches, self.val_groundtruth_patches, self.val_narrowband_patches,
                            None, None, None)

    def __getitem__(self, index):
        patch_img = torch.from_numpy(self.img_patches[index]).float()
        patch_groundtruth = torch.from_numpy(self.groundtruth_patches[index]).float()
        try:
            patch_narrowband = torch.from_numpy(self.narrowband_patches[index]).float()
            return patch_img, patch_groundtruth, patch_narrowband
        except AttributeError:
            return patch_img, patch_groundtruth

    def __len__(self):
        return self.data_len


if __name__ == "__main__":
    DRIVE_train = DRIVEDataset(
        "train",
        "../DRIVE_datasets_training_testing/DRIVE_dataset_imgs_train.hdf5",
        "../DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_train.hdf5",
        "../DRIVE_datasets_training_testing/DRIVE_dataset_narrowBand_train.hdf5",
        48,
        48,
        190000,
    )

    DRIVE_val = DRIVE_train.get_validation_dataset()

    DRIVE_test = DRIVEDataset(
        "test",
        "../DRIVE_datasets_training_testing/DRIVE_dataset_imgs_test.hdf5",
        "../DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_test.hdf5",
        "../DRIVE_datasets_training_testing/DRIVE_dataset_narrowBand_test.hdf5",
        48,
        48,
        190000,
    )

    print(len(DRIVE_train))

    # imga, msk = DRIVE_train.__getitem__(0)
    # print(imga.shape)
    # print(msk.shape)

    # imga, msk, narr = DRIVE_val.__getitem__(0)
    # print(imga.shape)
    # print(msk.shape)
    # print(narr.shape)

    # imga, msk = DRIVE_test.__getitem__(0)
    # print(imga.shape)
    # print(msk.shape)
