import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['lr'][idx] / 255., f['hr'][idx] / 255.

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

class ValDataset(Dataset):
    def __init__(self, h5_file):
        super(ValDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['lr'][idx] / 255., f['hr'][idx] / 255.

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


def training_data(train_path_X, train_path_Y, out_path):
    with h5py.File(out_path, "w") as f:
        low_res_images = []
        high_res_images = []
        for filename in sorted(os.listdir(train_path_X)):
            low_res_image = cv2.imread(os.path.join(train_path_X, filename))
            low_res_image = cv2.cvtColor(low_res_image, cv2.COLOR_BGR2RGB)
            high_res_image = cv2.imread(os.path.join(train_path_Y, filename))
            high_res_image = cv2.cvtColor(high_res_image, cv2.COLOR_BGR2RGB)

            low_res_image = np.array(low_res_image).astype(np.float32).transpose(2,0,1)
            high_res_image = np.array(high_res_image).astype(np.float32).transpose(2,0,1)

            low_res_images.append(low_res_image)
            high_res_images.append(high_res_image)
        
        low_res_images = np.array(low_res_images)
        high_res_images = np.array(high_res_images)
        f.create_dataset("lr", data=low_res_images)
        f.create_dataset("hr", data=high_res_images)


def val_data(train_path_X, train_path_Y, out_path):
    with h5py.File(out_path, "w") as f:
        low_res_images = []
        high_res_images = []
        for i, filename in enumerate(sorted(os.listdir(train_path_X))):
            low_res_image = cv2.imread(os.path.join(train_path_X, filename))
            low_res_image = cv2.cvtColor(low_res_image, cv2.COLOR_BGR2RGB)
            high_res_image = cv2.imread(os.path.join(train_path_Y, filename))
            high_res_image = cv2.cvtColor(high_res_image, cv2.COLOR_BGR2RGB)

            low_res_image = np.array(low_res_image).astype(np.float32).transpose(2,0,1)
            high_res_image = np.array(high_res_image).astype(np.float32).transpose(2,0,1)

            low_res_images.append(low_res_image)
            high_res_images.append(high_res_image)
        
        low_res_images = np.array(low_res_images)
        high_res_images = np.array(high_res_images)
        f.create_dataset("lr", data=low_res_images)
        f.create_dataset("hr", data=high_res_images)

def generate_training_data(data_path, out_path, scale):

    if(not os.path.exists(out_path)):
        os.makedirs(out_path)

    c = 0
    if len(os.listdir(out_path)) > 0: c += len(os.listdir(out_path))
    for folder, j in zip(sorted(os.listdir(data_path)), range(10)):
        for image in sorted(os.listdir(os.path.join(data_path, folder))):
            image_path = os.path.join(data_path, folder, image)
            resized_image = resize_image(image_path, scale)
            plt.imsave(f'{out_path}{c}.png', resized_image)
            c = c+1

def generate_validation_set(data_path_X, out_path_X, data_path_Y, out_path_Y, scale):

    if(not os.path.exists(out_path_X)):
        os.makedirs(out_path_X)

    c = 0
    for folder, j in zip(sorted(os.listdir(data_path_X)), range(10)):
        for image in sorted(os.listdir(os.path.join(data_path_X, folder))):
            image_path = os.path.join(data_path_X, folder, image)
            resized_image = resize_image(image_path, scale)
            plt.imsave(f'{out_path_X}{c}.png', resized_image)
            c = c+1

    if(not os.path.exists(out_path_Y)):
        os.makedirs(out_path_Y)

    c = 0
    for folder, j in zip(sorted(os.listdir(data_path_Y)), range(10)):
        for image in sorted(os.listdir(os.path.join(data_path_Y, folder))):
            image_path = os.path.join(data_path_Y, folder, image)
            resized_image = resize_image(image_path, scale)
            plt.imsave(f'{out_path_Y}{c}.png', resized_image)
            c = c+1

def resize_image(img_path, scale):
    image = cv2.imread(img_path)
    resized_image = cv2.resize(image, dsize=(image.shape[1]//scale, image.shape[0]//scale), interpolation=cv2.INTER_CUBIC)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    return resized_image

