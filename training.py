import argparse
import numpy as np
import os
import torch

from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
import torch.backends.cudnn as cudnn

from dataset import TrainDataset, ValDataset
from model import ELSR
from preprocessing import psnr


def train(model, dataloader, loss_fn, optimizer, device, scheduler):
    model.train()
    train_loss = 0
    for i, data in enumerate(dataloader):
        lr, hr = data
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        np.save("plot_data/lr.npy", lr[0].cpu().numpy().transpose(1,2,0))
        np.save("plot_data/hr.npy", hr[0].cpu().numpy().transpose(1,2,0))
        np.save("plot_data/sr.npy", sr[0].detach().cpu().numpy().transpose(1,2,0))
        loss = loss_fn(sr, hr)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(dataloader)
    return avg_train_loss

def validate(model, dataloader, device):
    model.eval()
    psnr_sum = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            lr, hr = data
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            psnr_sum += psnr(sr, hr)

    avg_psnr = psnr_sum / len(dataloader)
    return avg_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True, help='Training dataset h5 file')
    parser.add_argument('--val', type=str, required=True, help="Validation dataset h5 file")
    parser.add_argument('--out', type=str, required=True, help="Checkpoint folder")
    parser.add_argument('--scale', type=int, required=True, help="Upscale factor")
    parser.add_argument('--weights', type=str, help="weights_checkpoint.pth")
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs to train")
    parser.add_argument("--loss", type=str, default='mse', help="Specify mae or mse, loss function")
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ELSR(upscale_factor=args.scale).to(device)
    criterion = nn.MSELoss() if args.loss == 'mse' else nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Learning Rate Scheduler
    lambda1 = lambda epoch: args.lr*(0.5**(epoch//(args.epochs/5*2)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda1)

    train_dataset = TrainDataset(args.train)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True)
    
    val_dataset = ValDataset(args.val)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False)

    best_psnr = 0.0
    train_losses = []
    psnrs = []

    for epoch in range(1, args.epochs+1):
        train_loss = train(model=model, dataloader=train_dataloader, loss_fn=criterion, optimizer=optimizer, device=device, scheduler=scheduler)
        val_psnr = validate(model=model, dataloader=val_dataloader, device=device)

        train_losses.append(train_loss)
        psnrs.append(val_psnr)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(args.out,f'best_X{args.scale}_model.pth'))

        print(f"Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss}, Validation PSNR: {val_psnr}")
    
    torch.save(model.state_dict(), os.path.join(args.out, f'epoch_{epoch}_X{args.scale}.pth'))
    np.save(f'plot_data/train_losses_X{args.scale}_{args.loss}.npy', np.array(train_losses, dtype=np.float32))
