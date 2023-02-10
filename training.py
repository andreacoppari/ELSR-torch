import argparse
import copy
import numpy as np
import os
import torch
from tqdm import tqdm

from torch import nn
from torch.utils.data.dataloader import DataLoader
import torch.backends.cudnn as cudnn

from dataset import TrainDataset, ValDataset
from model import ELSR, AverageMeter
from preprocessing import psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--val', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--weights', type=str)
    args = parser.parse_args()

    EPOCHS = 500
    BATCH_SIZE = 16

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(123)

    model = ELSR(upscale_factor=4).to(device)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam([
        {'params': model.first_part.parameters()},
        {'params': model.last_part.parameters(), 'lr': 5e-5}
    ], lr=5e-4)

    # Learning Rate Scheduler
    lambda1 = lambda epoch: 5e-4*0*5**(epoch//200)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda1)

    train_dataset = TrainDataset(args.train)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  pin_memory=True)
    
    val_dataset = ValDataset(args.val)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % BATCH_SIZE), ncols=80) as t:
            t.set_description(f'epoch: {epoch+1}/{EPOCHS}')

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)
                loss = loss(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                t.set_postfix(loss=f'{epoch_losses.avg:.6f}')
                t.update(len(inputs))
    
    torch.save(model.state_dict(), os.path.join(args.out, f'epoch_{epoch}.pth'))

    model.eval()
    epoch_psnr = AverageMeter()

    for data in val_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)

        epoch_psnr.update(psnr(preds, labels), len(inputs))

    print(f'validation psnr: {epoch_psnr.avg:.2f}')

    if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print(f'best epoch: {best_epoch}, psnr: {best_psnr:.2f}')
    torch.save(best_weights, os.path.join(args.out, 'best.pth'))
