import torch

# 設定儲存模型檔案名字
model_file = 'model_CNN.pth'


class CnnModel(torch.nn.Module):
    def __init__(self):
        # 定義模型所需演算法
        super(CnnModel, self).__init__()

        # Convolution 1
        # input_shape = (3, 224, 224)
        # (weigh - kernel + 1) / stride = ( 224 - 5 + 1) / 1 = 220
        # output_shape = (16, 220, 220)
        # ref: https://reurl.cc/Y97oqO
        # ref: https://github.com/vdumoulin/conv_arithmetic
        self.cnn1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0
        )
        # activation
        # ReLU 會將 0 以下的值設為 0
        # ref: https://reurl.cc/QjnoEo
        # ref: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        self.relu1 = torch.nn.ReLU()
        # Max pool 1
        # 220 / 2 = 110
        # output_shape=(16, 110, 110)
        # ref: https://reurl.cc/dXzo12
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        # input_shape = (16, 110, 110)
        # (weigh - kernel + 1) / stride = ( 110 - 11 + 1) / 1 = 100
        # output_shape = (8, 100, 100)
        self.cnn2 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=8,
            kernel_size=11,
            stride=1,
            padding=0
        )
        self.relu2 = torch.nn.ReLU()
        # Max pool 2
        # weight / kernel = 100 / 2 = 50
        # output_shape = (8, 50, 50)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)

        # Fully connected 1
        # input_shape = (8 * 50 * 50)
        # ouput_shape = (512)
        # ref: https://reurl.cc/8W4VXd
        self.fc1 = torch.nn.Linear(8 * 50 * 50, 512)
        self.relu5 = torch.nn.ReLU()

        # Fully connected 2
        # input_shape = (512)
        # ouput_shape = (2)
        self.fc2 = torch.nn.Linear(512, 2)

    def forward(self, x):
        # 定義模型每一步的執行方式
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        # 將資料從三維轉成一維給 Linear 函數
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)

        return out