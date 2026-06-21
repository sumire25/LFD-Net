import os
import csv
import torch
import torch.nn as nn
import torchvision
import torch.optim
import argparse
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from pytorch_msssim import ssim

import image_data_loader
import model


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def calculate_sam(img1, img2):
    eps = 1e-8
    dot = (img1 * img2).sum(dim=1)
    norm1 = img1.pow(2).sum(dim=1).sqrt()
    norm2 = img2.pow(2).sum(dim=1).sqrt()
    cos = dot / (norm1 * norm2 + eps)
    cos = cos.clamp(-1, 1)
    sam = torch.acos(cos).mean()
    return torch.degrees(sam).item()


def calculate_ergas(img1, img2, ratio=1):
    mean_ref = img2.mean(dim=(2, 3), keepdim=True)
    rmse = (img1 - img2).pow(2).mean(dim=(2, 3), keepdim=True).sqrt()
    ergas = 100 / ratio * ((rmse / (mean_ref + 1e-8)).pow(2).mean(dim=1)).sqrt().mean()
    return ergas.item()


def validate(lfd_net, validation_data_loader, val_freq, epoch):
    lfd_net.eval()
    psnr_list, ssim_list, sam_list, ergas_list = [], [], [], []

    with torch.no_grad():
        for hazefree_image, hazy_image in validation_data_loader:
            hazefree_image = hazefree_image.cuda()
            hazy_image = hazy_image.cuda()
            dehaze_image = lfd_net(hazy_image).clamp_(0, 1)

            mse_val = F.mse_loss(dehaze_image, hazefree_image)
            psnr_val = 10 * torch.log10(1 / mse_val).item()
            psnr_list.append(psnr_val)

            _, _, H, W = dehaze_image.size()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(F.adaptive_avg_pool2d(dehaze_image, (int(H / down_ratio), int(W / down_ratio))),
                            F.adaptive_avg_pool2d(hazefree_image, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False).item()
            ssim_list.append(ssim_val)

            sam_val = calculate_sam(dehaze_image, hazefree_image)
            sam_list.append(sam_val)

            ergas_val = calculate_ergas(dehaze_image, hazefree_image)
            ergas_list.append(ergas_val)

    lfd_net.train()
    return np.mean(psnr_list), np.mean(ssim_list), np.mean(sam_list), np.mean(ergas_list)


def train(args):
    lfd_net = model.LFD_Net().cuda()
    lfd_net.apply(weights_init)

    training_data = image_data_loader.Haze1kDataset(args["train_hazy"], args["train_original"])
    validation_data = image_data_loader.Haze1kDataset(args["val_hazy"], args["val_original"])

    training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    print("Number of Training Images:", len(training_data))
    print("Number of Validation Images:", len(validation_data))

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(lfd_net.parameters(), lr=float(args["learning_rate"]), weight_decay=0.0001)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    lfd_net.train()

    val_freq = int(args.get("val_freq", 3))
    save_dir = args.get("save_dir", "trained_weights")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("training_data_captures", exist_ok=True)

    csv_path = os.path.join(save_dir, 'training_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_psnr', 'val_ssim', 'val_sam', 'val_ergas'])

    best_psnr = 0
    num_of_epochs = int(args["epochs"])
    for epoch in range(num_of_epochs):
        epoch_loss = 0
        num_batches = 0
        for iteration, (hazefree_image, hazy_image) in enumerate(training_data_loader):
            hazefree_image = hazefree_image.cuda()
            hazy_image = hazy_image.cuda()
            dehaze_image = lfd_net(hazy_image)
            loss = criterion(dehaze_image, hazefree_image)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lfd_net.parameters(), 0.1)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            num_batches += 1
            if ((iteration + 1) % 10) == 0:
                print("Epoch", epoch + 1, "Loss at iteration", iteration + 1, ":", loss.item())

        avg_loss = epoch_loss / max(num_batches, 1)

        if (epoch + 1) % val_freq == 0 or epoch == 0:
            avg_psnr, avg_ssim, avg_sam, avg_ergas = validate(lfd_net, validation_data_loader, val_freq, epoch)
            print(f"[Val Epoch {epoch+1}] PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | SAM: {avg_sam:.4f} | ERGAS: {avg_ergas:.4f}")

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_loss, avg_psnr, avg_ssim, avg_sam, avg_ergas])

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(lfd_net.state_dict(), os.path.join(save_dir, "best.pth"))

        torch.save(lfd_net.state_dict(), os.path.join(save_dir, "Epoch_" + str(epoch) + '.pth'))

    print(f"Training finished. Best PSNR: {best_psnr:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-th", "--train_hazy", required=True, help="path to hazy training images (input)")
    ap.add_argument("-to", "--train_original", required=True, help="path to original training images (target)")
    ap.add_argument("-vh", "--val_hazy", required=True, help="path to hazy validation images (input)")
    ap.add_argument("-vo", "--val_original", required=True, help="path to original validation images (target)")
    ap.add_argument("-e", "--epochs", required=True, help="number of epochs for training")
    ap.add_argument("-lr", "--learning_rate", required=True, help="learning rate for training")
    ap.add_argument("--val_freq", default="3", help="validation frequency in epochs")
    ap.add_argument("--save_dir", default="trained_weights", help="directory to save weights")

    args = vars(ap.parse_args())

    train(args)
