import os
import numpy as np
import matplotlib.pyplot as plt

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


def training_validation_split(dir, train_list: list, val_list: list):
    for folder in sorted(os.listdir(dir)):
        f_path = os.path.join(dir, folder)
        if os.path.isdir(f_path):
            images = []
            for image_name in sorted(os.listdir(f_path)):
                image_path = os.path.join(f_path, image_name)
                image = plt.imread(image_path)
                images.append(image)
            train_list.append(images[:210])
            val_list.append(images[210:])


def test_data(dir, test_list: list):
    for folder in sorted(os.listdir(dir)):
        folder_path = os.path.join(dir, folder)
        if os.path.isdir(folder_path):
            images = []
            for image_name in sorted(os.listdir(folder_path)):
                image_path = os.path.join(folder_path, image_name)
                image = plt.imread(image_path)
                images.append(image)
            test_list.append(images)


def get_training_data():
    # train set
    train_X = []
    train_Y = []

    # val set
    val_X = []
    val_Y = []

    # Load training/validation data
    training_validation_split(train_sharp_bicubic, train_X, val_X)
    print("DONE 1")
    training_validation_split(train_blur_bicubic, train_X, val_X)
    print("DONE 2")
    training_validation_split(train_sharp, train_Y, val_Y)
    print("DONE 3")
    training_validation_split(train_blur, train_Y, val_Y)
    print("DONE 4")

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    val_X = np.array(val_X)
    val_Y = np.array(val_Y)

    return train_X, train_Y, val_X, val_Y

def get_test_data():
    # test set
    test_X = []
    test_Y = []

    # Load test data
    test_data(val_sharp_bicubic, test_X)
    test_data(val_blur_bicubic, test_X)
    test_data(val_sharp, test_Y)
    test_data(val_blur, test_Y)

    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    return test_X, test_Y

# # Save the data splits as .npy files
# np.save('train_X.npy', train_X)
# np.save('train_Y.npy', train_Y)

# np.save('val_X.npy', val_X)
# np.save('val_Y.npy', val_Y)

# np.save('test_X.npy', test_X)
# np.save('test_Y.npy', test_Y)
