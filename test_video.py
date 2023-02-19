import argparse
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from PIL import Image
from model import ELSR
from preprocessing import psnr, prepare_img
from time import time

def test_video(model, device, video, upscale_factor):
    lr_video = []
    bicubic_video = []
    vid = []
    sr_video = []
    for img in video:
        img = cv2.resize(img, (img.shape[1]//upscale_factor*upscale_factor, img.shape[0]//upscale_factor*upscale_factor), interpolation=cv2.INTER_CUBIC)
        lr_img = cv2.resize(img, (img.shape[1]//upscale_factor, img.shape[0]//upscale_factor), interpolation=cv2.INTER_CUBIC)
        bicubic_upscaled = cv2.resize(lr_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        lr_img = prepare_img(lr_img, device)
        bicubic_upscaled = prepare_img(bicubic_upscaled, device)
        img = prepare_img(img, device)

        lr_video.append(lr_img)
        bicubic_video.append(bicubic_upscaled)
        vid.append(img)

    with torch.no_grad():
        t0 = time()
        for lr in lr_video:
            sr_img = model(lr).clamp(0, 1)
            sr_video.append(sr_img)
        t = time() - t0

    return sr_video, bicubic_video, vid, t

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--scale', type=int, required=True)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ELSR(upscale_factor=args.scale).to(device)

    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict=state_dict)
    model.eval()

    video = []
    for frame_path in os.listdir(args.input):
        frame = cv2.cvtColor(cv2.imread(os.path.join(args.input, frame_path)), cv2.COLOR_BGR2RGB)
        video.append(frame)

    sr_video, bicubic_video, video, t = test_video(model, device, video, upscale_factor=args.scale)

    avg_psnr = 0.0
    for sr_img, image in zip(sr_video, video):
        avg_psnr += psnr(sr_img, image)
    avg_psnr /= len(sr_video)

    bicubic_psnr = 0.0
    for bicubic_image, image in zip(bicubic_video, video): 
        bicubic_psnr += psnr(bicubic_image, image)
    bicubic_psnr /= len(bicubic_video)

    print(f"PSNR of Bicubic upscaled: {bicubic_psnr} dB")
    print(f"PSNR of Super-resoluted video: {avg_psnr} dB")
    print(f'FPS: {1/(t/100):.1f}')

    #Save videos
    
    gif = []
    for i, sr_img in enumerate(sr_video):
        out = sr_img.cpu().numpy().squeeze(0).transpose(1, 2, 0)
        plt.imsave(f"out/sr_video/sr_{i}.png", out)
        gif.append(Image.fromarray((out*255).astype(np.uint8)))
    gif[0].save('out/sr_video.gif', save_all=True, append_images=gif[1:], duration=10, loop=0)

    gif = []
    for i, sr_img in enumerate(sr_video):
        out = sr_img.cpu().numpy().squeeze(0).transpose(1, 2, 0)
        plt.imsave(f"out/bicubic_video/bicubic_{i}.png", out)
        gif.append(Image.fromarray((out*255).astype(np.uint8)))
    gif[0].save('out/bicubic_video.gif', save_all=True, append_images=gif[1:], duration=10, loop=0)

