import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import jaccard_score, f1_score
from unet.unet_parts import *

# 定义你的模型结构（与训练时使用的结构相同）
class YourModel(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_prob=0.2, bilinear=True):
        super(YourModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (DEFDown(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (DEFDown(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (DEFDown(512, 1024 // factor))

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

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
    n_channels = 3  # 输入通道数，例如RGB图像为3
    n_classes = 1  # 确保与训练时使用的类别数一致
    model = YourModel(n_channels, n_classes)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # 过滤掉不匹配的键
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # 增加batch维度
    return image

def calculate_metrics(pred_mask, true_mask):
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()
    iou = jaccard_score(true_mask, pred_mask, average='binary', zero_division=0)
    dice = f1_score(true_mask, pred_mask, average='binary', zero_division=0)
    return iou, dice

def save_prediction(pred_mask, output_path):
    pred_image = Image.fromarray((pred_mask * 255).astype(np.uint8))  # 将二值mask转换为图像
    pred_image.save(output_path)

def main():
    model_path = 'C:/Users/gr0665hh/Desktop/Pytorch-UNet-master/Pytorch-UNet-master/checkpoints/checkpoint_epoch451.pth'
    original_dir = 'C:/Users/gr0665hh/Desktop/Pytorch-UNet-master/Pytorch-UNet-master/data/imgs'
    mask_dir = 'C:/Users/gr0665hh/Desktop/Pytorch-UNet-master/Pytorch-UNet-master/data/masks'
    output_dir = 'C:/Users/gr0665hh/Desktop/Pytorch-UNet-master/predictions'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    model = load_model(model_path)

    total_iou = 0
    total_dice = 0
    num_images = 0

    for filename in os.listdir(original_dir):
        if filename.endswith('.tif'):
            original_image_path = os.path.join(original_dir, filename)
            mask_filename = filename.replace('.tif', '_mask.tif')
            mask_image_path = os.path.join(mask_dir, mask_filename)
            output_path = os.path.join(output_dir, filename)

            if not os.path.exists(mask_image_path):
                print(f"Mask for {filename} does not exist.")
                continue

            original_image = preprocess_image(original_image_path, transform)
            true_mask = Image.open(mask_image_path).convert('L')
            true_mask = true_mask.resize((256, 256))
            true_mask = np.array(true_mask) // 255

            with torch.no_grad():
                output = model(original_image)
                pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
                pred_mask = (pred_mask > 0.5).astype(np.uint8)

            true_mask = true_mask.flatten()
            pred_mask = pred_mask.flatten()

            print(f"True mask shape: {true_mask.shape}, Predicted mask shape: {pred_mask.shape}")

            iou, dice = calculate_metrics(pred_mask, true_mask)
            total_iou += iou
            total_dice += dice
            num_images += 1

            print(f'{filename} - IOU: {iou:.4f}, Dice: {dice:.4f}')

            # 保存预测的图像
            save_prediction(pred_mask.reshape(256, 256), output_path)

    if num_images > 0:
        average_iou = total_iou / num_images
        average_dice = total_dice / num_images
        print(f'Average IOU: {average_iou:.4f}')
        print(f'Average Dice: {average_dice:.4f}')
    else:
        print("No images found for processing.")

if __name__ == '__main__':
    main()
