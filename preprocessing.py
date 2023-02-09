import cv2
import random
import numpy as np

def resize_img(image):
    return cv2.resize(image, (180, 320))

def augment_data(low_res, high_res):
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

def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    if dim_order == 'hwc':
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
        cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    else:
        y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
        cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
        cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])

def convert_ycbcr_to_rgb(img, dim_order='hwc'):
    if dim_order == 'hwc':
        r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
        g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
        b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    else:
        r = 298.082 * img[0] / 256. + 408.583 * img[2] / 256. - 222.921
        g = 298.082 * img[0] / 256. - 100.291 * img[1] / 256. - 208.120 * img[2] / 256. + 135.576
        b = 298.082 * img[0] / 256. + 516.412 * img[1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])
