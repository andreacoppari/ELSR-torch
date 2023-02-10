import argparse
import cv2
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from model import ELSR
from preprocessing import convert_ycbcr_to_rgb, psnr, prepare_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = ELSR(upscale_factor=4).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))


    lr, _ = prepare_img(lr, device)
    hr, _ = prepare_img(hr, device)
    _, ycbcr = prepare_img(bicubic, device)

    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)

    image_psnr = psnr(hr, preds)
    print(f'PSNR: {image_psnr:.2f}')

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', '_espcn_x{}.'.format(args.scale)))
