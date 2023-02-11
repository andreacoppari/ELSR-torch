import argparse
import matplotlib.pyplot as plt
import cv2

import torch
import torch.backends.cudnn as cudnn

from model import ELSR
from preprocessing import psnr, prepare_img

def test_image(model, device, img, upscale_factor):
    lr_img = cv2.resize(img, (img.shape[1]//upscale_factor, img.shape[0]//upscale_factor), interpolation=cv2.INTER_CUBIC)
    bicubic_upscaled = cv2.resize(lr_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    lr_img = prepare_img(lr_img, device)
    bicubic_upscaled = prepare_img(bicubic_upscaled, device)
    img = prepare_img(img, device)

    with torch.no_grad():
        sr_img = model(lr_img)

    return sr_img, bicubic_upscaled, img

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

    image = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)

    sr_img, bicubic_upscaled, image = test_image(model, device, image, upscale_factor=args.scale)

    psnr_value = psnr(sr_img, image)
    bicubic_psnr = psnr(bicubic_upscaled, image)
    print(f"PSNR of Bicubic upscaled: {bicubic_psnr} dB")
    print(f"PSNR of Super-resoluted image: {psnr_value} dB")

    #Save images
    
    out = sr_img.cpu().numpy().squeeze(0).transpose(1,2,0)
    bicubic_out = bicubic_upscaled.cpu().numpy().squeeze(0).transpose(1, 2, 0)

    plt.imsave("out/output.png", out)
    plt.imsave("out/bicubic.png", bicubic_out)
