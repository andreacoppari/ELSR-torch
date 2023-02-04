import cv2
import random

def resize_img(image):
    return cv2.resize(image, (320, 180))

def augment_data(low_res, high_res):
    # Randomly choose a type of augmentation
    aug_type = random.choice(["flip", "rotate", "zoom", "none"])
    
    # Perform the chosen type of augmentation
    if aug_type == "flip":
        low_res = cv2.flip(low_res, 1)
        high_res = cv2.flip(high_res, 1)
    elif aug_type == "rotate":
        angle = random.uniform(-30, 30)
        rows, cols = low_res.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        low_res = cv2.warpAffine(low_res, M, (cols, rows))
        high_res = cv2.warpAffine(high_res, M, (cols, rows))
    elif aug_type == "zoom":
        zoom_scale = random.uniform(0.8, 1.2)
        rows, cols = low_res.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 0, zoom_scale)
        low_res = cv2.warpAffine(low_res, M, (cols, rows))
        high_res = cv2.warpAffine(high_res, M, (cols, rows))
    
    return low_res, high_res
