import cv2
import random
import numpy as np
import torch

def augment_data(low_res, high_res):
    # Read images
    low_res = cv2.cvtColor(cv2.imread(low_res), cv2.COLOR_BGR2RGB)
    high_res = cv2.cvtColor(cv2.imread(high_res), cv2.COLOR_BGR2RGB)

    # Randomly choose a type of augmentation
    aug_type = random.choice(["flip", "rotate", "zoom", "none"])

    # Perform the chosen type of augmentation
    if aug_type == "flip":
        low_res = cv2.flip(low_res, 1)
        high_res = cv2.flip(high_res, 1)
    elif aug_type == "rotate":
        angle = random.uniform(-30, 30)
        rowsLR, colsLR = low_res.shape[:2]
        MLR = cv2.getRotationMatrix2D((colsLR/2, rowsLR/2), angle, 1)
        low_res = cv2.warpAffine(low_res, MLR, (colsLR, rowsLR))
        rowsHR, colsHR = high_res.shape[:2]
        MHR = cv2.getRotationMatrix2D((colsHR/2, rowsHR/2), angle, 1)
        high_res = cv2.warpAffine(high_res, MHR, (colsHR, rowsHR))
    elif aug_type == "zoom":
        zoom_scale = random.uniform(0.8, 1.2)
        rowsLR, colsLR = low_res.shape[:2]
        MLR = cv2.getRotationMatrix2D((colsLR/2, rowsLR/2), 0, zoom_scale)
        low_res = cv2.warpAffine(low_res, MLR, (colsLR, rowsLR))
        rowsHR, colsHR = high_res.shape[:2]
        MHR = cv2.getRotationMatrix2D((colsHR/2, rowsHR/2), 0, zoom_scale)
        high_res = cv2.warpAffine(high_res, MHR, (colsHR, rowsHR))
    
    return low_res, high_res

def psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def prepare_img(img, device):
    x = np.array(img, dtype='float32') / 255.
    x = torch.from_numpy(x).to(device)
    x = x.permute(2, 0, 1).unsqueeze(0)  # permute the dimensions to match the input size of the model
    return x
