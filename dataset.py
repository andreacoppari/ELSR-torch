import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset
from preprocessing import convert_rgb_to_y


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

class ValDataset(Dataset):
    def __init__(self, h5_file):
        super(ValDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


def training_data(out_path):
    h5_file = h5py.File(out_path, "w")

    lr_patches = []
    hr_patches = []

    for image_path in sorted(os.listdir("datasets/resized/aug")):
        lr = cv2.cvtColor(cv2.imread(f"datasets/resized/aug/{image_path}"), cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(cv2.imread(f"datasets/sr_resized/aug/{image_path}"), cv2.COLOR_BGR2RGB)

        lr = np.array(lr, dtype='float32')
        hr = np.array(hr, dtype='float32')
        lr = convert_rgb_to_y(lr)
        hr = convert_rgb_to_y(hr)

        lr_patches.append(lr)
        hr_patches.append(hr)
    
    h5_file.create_dataset('lr', data=np.array(lr_patches))
    h5_file.create_dataset('hr', data=np.array(hr_patches))

    h5_file.close()

def val_data(out_path):
    h5_file = h5py.File(out_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(os.listdir("datasets/val_resized/"))):
        lr = cv2.cvtColor(cv2.imread(f"datasets/val_resized/{image_path}"), cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(cv2.imread(f"datasets/val_sr_resized/{image_path}"), cv2.COLOR_BGR2RGB)

        lr = np.array(lr, dtype='float32')
        hr = np.array(hr, dtype='float32')
        lr = convert_rgb_to_y(lr)
        hr = convert_rgb_to_y(hr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


def generate_training_data(data_path):
    c = 0
    if len(os.listdir('datasets/resized/')) > 0: c += 1000
    for folder, j in zip(sorted(os.listdir(data_path)), range(10)):
        for image in sorted(os.listdir(os.path.join(data_path, folder))):
            image_path = os.path.join(data_path, folder, image)
            resized_image = resize_image(image_path)
            plt.imsave(f'datasets/resized/{c}.png', resized_image)
            c = c+1

def generate_label_data(data_path):
    c = 0
    if len(os.listdir('datasets/sr_resized/')) > 0: c += 1000
    for folder, j in zip(sorted(os.listdir(data_path)), range(10)):
        for image in sorted(os.listdir(os.path.join(data_path, folder))):
            image_path = os.path.join(data_path, folder, image)
            resized_image = resize_image(image_path)
            plt.imsave(f'datasets/sr_resized/{c}.png', resized_image)
            c = c+1

def generate_validation_set(data_path_X, data_path_Y):
    c = 0
    for folder, j in zip(sorted(os.listdir(data_path_X)), range(10)):
        for image in sorted(os.listdir(os.path.join(data_path_X, folder))):
            image_path = os.path.join(data_path_X, folder, image)
            resized_image = resize_image(image_path)
            plt.imsave(f'datasets/val_resized/{c}.png', resized_image)
            c = c+1
    c = 0
    for folder, j in zip(sorted(os.listdir(data_path_Y)), range(10)):
        for image in sorted(os.listdir(os.path.join(data_path_Y, folder))):
            image_path = os.path.join(data_path_Y, folder, image)
            resized_image = resize_image(image_path)
            plt.imsave(f'datasets/val_sr_resized/{c}.png', resized_image)
            c = c+1

def resize_image(img_path):
    image = cv2.imread(img_path)
    resized_image = cv2.resize(image, dsize=(image.shape[1]//4, image.shape[0]//4))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    return resized_image

