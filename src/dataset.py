
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from preprocess.extract_patches import get_data_training, get_data_testing


class SSTemDataset(Dataset):
    pass


class DRIVEDataset(Dataset):

    def __init__(self, mode, img_path, groudtruth_path, height, width, n_subimgs, inside_fov=True):
        if mode == 'train':
            self.img_patches, self.groundtruth_patches = get_data_training(img_path, groudtruth_path,
                                                                           height, width, n_subimgs, inside_fov)
        elif mode == 'test':
            self.img_patches, self.groundtruth_patches = get_data_testing(img_path, groudtruth_path,
                                                                          n_subimgs, height, width)
        else:
            raise ValueError("mode needs to be train | test")
        self.data_len = len(self.img_patches)

    def __getitem__(self, index):
        patch_img = torch.from_numpy(self.img_patches[index]).float()
        patch_groundtruth = torch.from_numpy(self.groundtruth_patches[index]).float()
        return patch_img, patch_groundtruth

    def __len__(self):
        return self.data_len


if __name__ == "__main__":

    DRIVE_train = DRIVEDataset(
        "train",
        "../DRIVE_datasets_training_testing/DRIVE_dataset_imgs_train.hdf5",
        "../DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_train.hdf5",
        48,
        48,
        190000
    )
    DRIVE_test = DRIVEDataset(
        "test",
        "../DRIVE_datasets_training_testing/DRIVE_dataset_imgs_test.hdf5",
        "../DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_test.hdf5",
        48,
        48,
        10000
    )

    imga, msk = DRIVE_train.__getitem__(0)
    print(imga.shape)
    print(msk.shape)

    imga, msk = DRIVE_test.__getitem__(0)
    print(imga.shape)
    print(msk.shape)


