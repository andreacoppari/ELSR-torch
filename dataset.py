import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py

root_dir = "./datasets/"
train_dir = os.path.join(root_dir, "train/")
val_dir = os.path.join(root_dir, "val/")

train_sharp = os.path.join(train_dir, "train_sharp/")
train_blur = os.path.join(train_dir, "train_blur/")
train_sharp_bicubic = os.path.join(train_dir, "train_sharp_bicubic/X4")
train_blur_bicubic = os.path.join(train_dir, "train_blur_bicubic/X4")

val_sharp = os.path.join(val_dir, "val_sharp/")
val_blur = os.path.join(val_dir, "val_blur/")
val_sharp_bicubic = os.path.join(val_dir, "val_sharp_bicubic/X4")
val_blur_bicubic = os.path.join(val_dir, "val_blur_bicubic/X4")


def training_data(dir, train_list: list):
    for folder in sorted(os.listdir(dir))[:4]:
        f_path = os.path.join(dir, folder)
        if os.path.isdir(f_path):
            images = []
            for image_name in sorted(os.listdir(f_path)):
                image_path = os.path.join(f_path, image_name)
                image = plt.imread(image_path)
                images.append(image)
            train_list.append(images)


def val_data(dir, val_list: list):
    for folder in sorted(os.listdir(dir))[4:5]:
        folder_path = os.path.join(dir, folder)
        if os.path.isdir(folder_path):
            images = []
            for image_name in sorted(os.listdir(folder_path)):
                image_path = os.path.join(folder_path, image_name)
                image = plt.imread(image_path)
                images.append(image)
            val_list.append(images)


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

    # # train set
    # train_X = []
    # train_Y = []

    # # val set
    # val_X = []
    # val_Y = []

    # # Load training/validation data
    # training_data(train_sharp_bicubic, train_X)
    # print("DONE trainX 1")
    # training_data(train_blur_bicubic, train_X)
    # print("DONE trainX 2")
    # training_data(train_sharp, train_Y)
    # print("DONE trainY 1")
    # training_data(train_blur, train_Y)
    # print("DONE trainY 2")

    # val_data(train_sharp_bicubic, val_X)
    # print("DONE valX 1")
    # val_data(train_blur_bicubic, val_X)
    # print("DONE valX 2")
    # val_data(train_sharp, val_Y)
    # print("DONE valY 1")
    # val_data(train_blur, val_Y)
    # print("DONE valY 2")

    # return train_X, train_Y, val_X, val_Y

def resize_image(img_path):
    image = cv2.imread(img_path)
    resized_image = cv2.resize(image, dsize=(image.shape[1]//4, image.shape[0]//4))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    return resized_image

# # Save the data splits as .npy files
# np.save('train_X.npy', train_X)
# np.save('train_Y.npy', train_Y)

# np.save('val_X.npy', val_X)
# np.save('val_Y.npy', val_Y)
