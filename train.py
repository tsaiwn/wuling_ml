import pathlib

import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm import tqdm

# 設定使用的模型及儲存的名字
from model import CnnModel as Model, model_file

# 設定每批丟入多少張圖片
BATCH_SIZE = 32
# 設定學習率
LEARNING_RATE = 0.0001
# 設定模型訓練次數
EPOCHS = 50

# 設定是否預覽圖片
preview_image = False
preview_model = False

# 設定檔案路徑
train_path = pathlib.Path("train")
valid_path = pathlib.Path("validation")

# 設定是否強制載入已存在的模型
force_load_exist_model = True

# 損失(loss)初始值 = 無限大，目標是儘可能小
valid_loss_min = np.Inf

# 讀取電腦是否支援 GPU 運算
train_on_gpu = torch.cuda.is_available()

# 若電腦支援 GPU 運算，所要使用的 GPU
cuda_device = torch.device()

# 1. 讀取圖片
print("1. 讀取圖片")

# 1.1 設定圖片轉換器，將圖檔 resize 至(224,224)像素，然後轉換成 tensor 的資料型態
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 1.2 從資料夾讀取圖片檔
train_data = datasets.ImageFolder(train_path, transform=image_transforms)
valid_data = datasets.ImageFolder(valid_path, transform=image_transforms)
print("\t模型分類分式，訓練資料：", train_data.class_to_idx)
print("\t模型分類分式，驗證資料：", valid_data.class_to_idx)

# 1.3 設定圖片載入器，讓後面的模型訓練可以批次執行
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)

# 1.4 預覽圖片
if preview_image:
    # 載入預覽模組
    import matplotlib.pyplot as plt

    # 設定分類
    classes = ['cat', 'dog']

    # 從圖片載入器取得圖片
    images, labels = iter(train_loader).next()

    # 設定顯示畫面大小
    fig = plt.figure(figsize=(60, 40))

    # 顯示圖片(20張)
    for idx in np.arange(20):
        # 設定圖片顯示位置
        ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
        # 轉換圖片呈可顯示模式 3x224x224 to 224x224x3
        image = images[idx].permute(1, 2, 0)
        plt.imshow(image)
        # 設定圖片標題
        ax.set_title(f"{classes[labels[idx]]}")

    plt.show()

# 2. 建立模型
print("2. 建立模型")

# 2.1 檢查是否支援使用 GPU 訓練
if not train_on_gpu:
    print("\t未支援 CUDA, 使用 CPU 進行訓練...")
else:
    print("\t支援 CUDA! 使用 GPU 進行訓練...")

# 2.2 建立模型
model = Model()

# 2.3 檢查是否有已存在的模型，進行載入
if pathlib.Path(model_file).is_file() and pathlib.Path(model_file + '.loss').is_file():
    if force_load_exist_model:
        print('\t發現已存在的模型，載入中...')
        model.load_state_dict(torch.load(model_file))
        with open(model_file + '.loss', 'r') as f:
            valid_loss_min = float(f.read())
    elif 'Y' in input('\t發現已存在的模型，是否載入？(y/n)').upper():
        print('\t載入模型...')
        model.load_state_dict(torch.load(model_file))
        with open(model_file + '.loss', 'r') as f:
            valid_loss_min = int(f.read())
    else:
        print('\t重新訓練新模型...')

# 2.4 預覽模型
if preview_model:
    if train_on_gpu:
        summary(model.cuda(), (3, 224, 224))
    else:
        summary(model.cpu(), (3, 224, 224))

# 3. 訓練模型
print("3. 訓練模型")

# 3.1 設定模型使用資源，CPU or GPU
if train_on_gpu:
    model.cuda(cuda_device)
else:
    model.cpu()

# 3.2 設定訓練所需資料
# 設定評斷(criterion)函數，判斷預測結果與實際值的接近程度
criterion = torch.nn.CrossEntropyLoss()
# 設定優化(optimizer)函數，用以調整模型的參數
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3.3 開始訓練
print(f'\t預計總訓練次數: {EPOCHS}')
for epoch in range(1, EPOCHS + 1):
    # 暫存當次訓練損失值
    train_loss = 0.0
    valid_loss = 0.0

    # 3.3.1 訓練階段
    # 將模型切換至訓練模式
    model.train()

    # 從圖片載入器取得圖片進行訓練
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, ncols=92, desc=f'\t第{epoch}次訓練')):
        # print(f'\t\t訓練中...第 {batch_idx} 批次')
        # 若可以使用 GPU，則使用 GPU
        if train_on_gpu:
            data, target = data.cuda(cuda_device), target.cuda(cuda_device)
        # 重設優化函數的梯度值
        optimizer.zero_grad()

        # 正向傳遞階段(forward pass):
        # 計算預測結果
        output = model(data)
        # 計算損失值
        loss = criterion(output, target)
        # 反向傳遞階段(backward pass):
        # 計算損失相對於模型參數的梯度
        loss.backward()
        # 執行優化步驟，更新模型參數
        optimizer.step()
        # 統計訓練損失值
        train_loss += loss.item() * data.size(0)

    # 3.3.2 驗證階段
    # 將模型切換至預測模式
    model.eval()
    for batch_idx, (data, target) in enumerate(tqdm(valid_loader, ncols=92, desc=f'\t第{epoch}次驗証')):
        # 若可以使用 GPU，則使用 GPU
        if train_on_gpu:
            data, target = data.cuda(cuda_device), target.cuda(cuda_device)

        # 正向傳遞階段(forward pass):
        # 計算預測結果
        output = model(data)
        # 計算損失值
        loss = criterion(output, target)
        # 驗証階段不進行更新模型，所以不需反向傳遞階段
        # 統計驗証損失值
        valid_loss += loss.item() * data.size(0)

    # 計算平均損失值
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)

    print(f'\tTraining Loss: {train_loss:.6f}')
    print(f'\tValidation Loss: {valid_loss:.6f}')

    # 若損失值比之前訓練的還小，代表模型比之前的還好，則儲存該模型
    if valid_loss < valid_loss_min:
        print('\tValidation loss decreased:')
        print(f'\t\t{valid_loss_min:.6f} -> {valid_loss:.6f}.')
        print('\t儲存模型...')
        torch.save(model.state_dict(), model_file)
        valid_loss_min = valid_loss
        with open(model_file + '.loss', 'w') as f:
            f.write(str(valid_loss_min))

print("訓練結束...")
