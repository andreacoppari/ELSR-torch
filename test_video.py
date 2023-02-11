import argparse
import matplotlib.pyplot as plt
import cv2
import os

import torch
import torch.backends.cudnn as cudnn

from model import ELSR
from preprocessing import psnr, prepare_img

def test_video(model, device, video, upscale_factor):
    lr_video = []
    bicubic_video = []
    vid = []
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
        for lr in lr_video:
            sr_img = model(lr).clamp(0, 1)
            sr_video.append(sr_img)

    return sr_video, bicubic_video, vid

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

    sr_video, bicubic_video, video = test_video(model, device, video, upscale_factor=args.scale)

    avg_psnr = 0.0
    for sr_img, image in zip(sr_video, video):
        avg_psnr += psnr(sr_img, image)
    avg_psnr /= len(sr_video)

    bicubic_psnr = 0.0
    for bicubic_image, image in zip(): 
        bicubic_psnr += psnr(bicubic_image, image)
    bicubic_psnr /= len(bicubic_video)

    print(f"PSNR of Bicubic upscaled: {bicubic_psnr} dB")
    print(f"PSNR of Super-resoluted image: {avg_psnr} dB")

    #Save images
    
    for i, sr_img in enumerate(sr_video):
        out = sr_img.cpu().numpy().squeeze(0).transpose(1, 2, 0)
        plt.imsave(f"out/sr_{i}.png", out)

    for i, sr_img in enumerate(sr_video):
        out = sr_img.cpu().numpy().squeeze(0).transpose(1, 2, 0)
        plt.imsave(f"out/bicubic_{i}.png", out)

