import os

import cv2
import torch
import torchvision.models as models
import numpy as np

def load_pth(pth_path):
    model = torch.load(pth_path)  # 加载模型
    model = model['model']  # 获取模型实例
    model.eval()  # 设置为评估模式
    return model

def cut_image(image, model):
    prediction = model(image)  # 调用模型
    prediction = prediction.argmax(1)  # 获取预测类别
    return prediction

def calculate_accuracy(prediction, mask):
    correct = np.sum(prediction == mask)
    total = mask.size
    accuracy = correct / total * 100
    return accuracy

def main():
    # 读取 pth 文件
    model = load_pth("checkpoints\checkpoint_epoch100.pth")

    # 获取 .tif 文件夹
    tif_folder = "C:\\MRITest_data\\images"

    # 获取 mask 文件夹
    mask_folder = "C:\\MRITest_data\\masks"

    # 初始化精度
    accuracy = 0.0

    # 遍历 .tif 文件
    for tif_file in os.listdir(tif_folder):
        # 读取 .tif 文件
        tif = cv2.imread(os.path.join(tif_folder, tif_file))

        # 将 .tif 文件转换为 torch 张量
        tif = torch.from_numpy(np.array(tif)).float()

        # 使用模型对 .tif 文件进行切割
        prediction = cut_image(tif, model)

        # 获取 mask 文件
        mask = cv2.imread(os.path.join(mask_folder, tif_file[:-4] + ".png"))

        # 计算精度
        accuracy += calculate_accuracy(prediction, mask)

    # 输出平均精度
    print("Average accuracy:", accuracy / len(os.listdir(tif_folder)))

if __name__ == "__main__":
    main()
