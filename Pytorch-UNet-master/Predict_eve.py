import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score
from unet.unet_parts import *
from utils.dice_score import multiclass_dice_coeff, dice_coeff

class YourModel(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_prob=0.2, bilinear=True):
        super(YourModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DEFDown(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = DEFDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DEFDown(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

def load_model(model_path):
    n_channels = 3
    n_classes = 2
    model = YourModel(n_channels, n_classes)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def save_prediction(pred_mask, output_path):
    pred_mask = pred_mask.squeeze(0)  # 去掉batch维度
    print(f"Pred mask after squeeze shape: {pred_mask.shape}")
    pred_mask = (pred_mask > 0.5).float()  # 二值化
    print(f"Pred mask after threshold: {pred_mask.unique()}")  # 打印唯一值检查二值化效果
    pred_mask = pred_mask.cpu().numpy()
    pred_image = Image.fromarray((pred_mask * 255).astype(np.uint8))  # 转换为uint8图像
    pred_image.save(output_path)

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for i, batch in enumerate(
                tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False)):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = net(image)

            if net.n_classes == 1:
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_true = mask_true.squeeze(1)  # 将 [batch_size, 1, height, width] 转换为 [batch_size, height, width]

                if mask_true.dim() == 4 and mask_true.size(1) == 1:
                    mask_true = mask_true.squeeze(1)  # 去掉维度

                if mask_true.dim() == 3:
                    mask_true = F.one_hot(mask_true, num_classes=net.n_classes).permute(0, 3, 1, 2).float()
                else:
                    raise RuntimeError(f"Unexpected mask_true dimensions: {mask_true.dim()}")

                mask_pred = F.one_hot(mask_pred.argmax(dim=1), num_classes=net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

                # 保存第一个预测结果以进行调试
                if i == 14:
                    save_prediction(mask_pred.argmax(dim=1), "output_path_here.png")
                    save_prediction(mask_true.argmax(dim=1), "true_mask_here.png")

    net.train()
    return dice_score / max(num_val_batches, 1)

def main():
    model_dir = 'C:/Users/gr0665hh/Desktop/Pytorch-UNet-master/Pytorch-UNet-master/checkpoints'
    original_dir = 'C:/Users/gr0665hh/Desktop/Pytorch-UNet-master/original_images'
    mask_dir = 'C:/Users/gr0665hh/Desktop/Pytorch-UNet-master/masks'
    output_dir = 'C:/Users/gr0665hh/Desktop/Pytorch-UNet-master/predictions'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = True

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(original_dir, mask_dir, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    best_dice_score = 0
    best_model_path = None

    for model_file in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_file)
        if model_path.endswith('.pth'):
            model = load_model(model_path).to(device)
            dice_score = evaluate(model, dataloader, device, amp)
            print(f'Model: {model_file}, Dice Score: {dice_score:.4f}')
            if dice_score > best_dice_score:
                best_dice_score = dice_score
                best_model_path = model_path

    print(f'Best Model: {best_model_path}, Dice Score: {best_dice_score:.4f}')

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [img for img in os.listdir(img_dir) if img.endswith('.tif')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.tif', '_mask.tif'))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = torch.unsqueeze(mask, 0)
        return {"image": image, "mask": mask}

if __name__ == '__main__':
    main()
