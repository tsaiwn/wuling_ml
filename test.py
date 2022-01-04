# -*- coding: utf-8 -*-
import pathlib

import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# 設定使用的模型及儲存的名字
from model import CnnModel as Model, model_file

# 設定檔案路徑
test_path = pathlib.Path("test")
# 設定每批丟入多少張圖片
BATCH_SIZE = 32

# 讀取電腦是否支援 GPU 運算
train_on_gpu = torch.cuda.is_available()

# 若電腦支援 GPU 運算，所要使用的 GPU
cuda_device = torch.device('cuda')

# 1. 讀取圖片
print("1. 讀取圖片")

# 1.1 設定圖片轉換器，將圖檔 resize 至(224,224)像素，然後轉換成 tensor 的資料型態
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 1.2 從資料夾讀取圖片檔
test_data = datasets.ImageFolder(test_path, transform=image_transforms)
print("\t模型分類分式，測試資料：", test_data.class_to_idx)

# 1.3 設定圖片載入器，讓後面的模型測試可以批次執行
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# 2. 載入模型
print("2. 載入模型")

# 2.1 建立模型
model = Model()

# 2.2 檢查是否支援使用 GPU 訓練
if not train_on_gpu:
    print("\t未支援 CUDA, 使用 CPU 進行測試...")
else:
    print("\t支援 CUDA! 使用 GPU 進行測試...")

# 2.3 檢查是否有已存在的模型，進行載入
if pathlib.Path(model_file).is_file():
    model.load_state_dict(torch.load(model_file))
else:
    print('找不到模型檔，中止測試')
    exit()

# 3. 測試模型
print("3. 測試模型")

# 3.1 設定模型使用資源，CPU or GPU
if train_on_gpu:
    model.to(cuda_device)
else:
    model.cpu()

# 3.2 宣告變數
# 設定評斷(criterion)函數，判斷預測結果與實際值的接近程度
criterion = torch.nn.CrossEntropyLoss()
# 暫存測試結果
test_loss = 0.  # 記錄總損失率
test_correct = 0.  # 記錄正確數
test_total = 0.  # 記錄總數

# 3.3 將模型切換至預測模式，進行預測
model.eval()
for batch_idx, (data, target) in enumerate(tqdm(test_loader, ncols=92, desc=f'\t測試進度')):
    # 若可以使用 GPU，則使用 GPU
    if train_on_gpu:
        data, target = data.to(cuda_device), target.cuda(cuda_device)

    # 正向傳遞階段(forward pass):
    # 計算預測結果
    output = model(data)
    # 計算損失值
    loss = criterion(output, target)

    # 更新測試損失值
    test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
    # 將預測結果 (0.xx, 0.yy) 轉換為 0 / 1，對應 cat / dog
    pred = output.data.max(1, keepdim=True)[1]
    # 計算預測正確數量
    test_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
    test_total += data.size(0)

# 輸出結果
accuracy = 100. * test_correct / test_total
print(f'\t損失值(loss): {test_loss:.6f}')
print(f'\t正確率(accuracy): {accuracy:.2f}%% ({test_correct:.2f}/{test_total:.2f})')
