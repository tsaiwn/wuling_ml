# 武陵高中 - 交大愛物 AI 課程

## 前言
本專案供武陵高中 110 學年度上學期所開設的 ***交大愛物 AI 課程*** 教學使用。專案內容為使用 pytorch 以 CNN 模型進行簡易的貓狗分類，其圖片來源使用了 [Kaggle](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) 上所提供的貓狗資料集。
由於為了教學使用，只從中取其貓狗各 1000 筆做訓練(training)使用、各 500 筆做驗証(validation)及各 500 筆做測試(testing)。(感謝顏泰翔博士Ksoy提供)

### 本專案請以 **python3.7** 以上版本執行。

### 因為 data 目錄內貓狗圖片兩萬多張若從 GitHub 下載總共有六百多MB, <br/>建議改到 https://goo.GL/6jtP41/ 點 00_50 進入後點 Day6 抓裡面的 .zip 檔案只有約75MB含程式碼和測試圖片。

## 安裝步驟
所需相關的套件已列在 `requirements.txt` 中，
請執行下述指令或依您所使用環境進行安裝相關套件。

    pip install -r requirements.txt

## 執行方式
1. 修改 `model.py`，調整 AI 模型結構 
2. 修改 `train.py`，調整訓練過程中的設定值
3. 執行訓練檔

    ```shell
    python train.py
    ```
4. 執行測試結果檔

    ```shell
    python test.py
    ```

## 說明
### 檔案結構
    .
    |-- README.md
    |-- model.py
    |-- requirements.txt
    |-- test
    |   |-- cat
    |   |   |-- cat.1051.jpg
    |   |   |-- ...
    |   |   `-- cat.2000.jpg
    |   `-- dog
    |       |-- dog.1501.jpg
    |       |-- ...
    |       `-- dog.2000.jpg
    |-- test.py
    |-- train
    |   |-- cat
    |   |   |-- cat.1.jpg
    |   |   |-- ...
    |   |   `-- cat.1000.jpg
    |   `-- dog
    |       |-- dog.1.jpg
    |       |-- ...
    |       `-- dog.1000.jpg
    |-- train.py
    `-- validation
        |-- cat
        |   |-- cat.1001.jpg
        |   |-- ...
        |   `-- cat.1500.jpg
        `-- dog
            |-- dog.1001.jpg
            |-- ...
            `-- dog.1500.jpg

### 資料集
本專案已包含訓練所需資料集

- `datas/original/` 原始資料集，含貓狗各 12500 張圖片
- `datas/1000/` 取原始資料貓狗各 1000 張圖片
- `datas/100/` 取原始資料貓狗各 100 張圖片
- `datas/test/` 供測試時使用，取貓狗各 500 張圖片

### 程式碼
本專案包含 3 個基本程式檔
- `model.py`
    此檔案定義了一個以 pytorch 套件所建構出的簡易 CNN 貓狗分類模型。其中包含一個變數 *`model_file`* 及一個模型類別 *`CnnModel`*
    - *`model_file`* 指定了訓練後的模型要儲存的檔名
    **注意**: 由於不同模型不能交叉使用，故若有修改模型就應以不同檔名儲存。
    - *`CnnModel`* 指定了所使用的 AI 模型，該模型繼承了 `torch.nn.Module`，其中必須覆寫二個函數分別為
        - `def __init__(self)` 定義了本模型將會使用哪些演算法
        - `def forward(self, x)` 定義了本模型的函數執行順序。

- `train.py` 
    此檔案提供了基本的訓練過程，其中包含了幾個重要參數的設定
    - `BATCH_SIZE` 指定了在模型訓練過程中，一次處理幾張圖片，可依據個人電腦狀況進行調整
    - `LEARNING_RATE` 指定了模型訓練過程中，每次優化的變化程度。值愈高變化愈大，在訓練前期可加速提高準確率；值愈小變化愈精細，在訓練後期可提高收歛程度。
    - `EPOCHS` 指定了模型訓練過程中，所訓練的次數，基本上需要一定的次數後模型才會有效預測，但隨著訓練次數增多，並不會更加準確。若要提高準確度，應試著調整模型或增加更多樣化的圖片

- `test.py`
    此檔案提供了訓練完的模型進行測試。

## 使用已訓練模型

`pretrained_model` 中包含已使用 12500 張貓狗圖片訓練了 50 次的模型。
可將其複制至根目錄進行測試，準確率應為 81%。

## 參考資料
1. https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
2. https://pytorch.org/docs/stable/index.html
3. https://docs.microsoft.com/zh-tw/windows/ai/windows-ml/tutorials/pytorch-train-model
5. https://github.com/pytorch/serve/tree/master/examples
6. https://github.com/vdumoulin/conv_arithmetic
7. https://medium.com/jimmyfu87/cnn%E5%AF%A6%E4%BD%9Ckaggle%E8%B2%93%E7%8B%97%E5%BD%B1%E5%83%8F%E8%BE%A8%E8%AD%98-pytorch-26380b357a3d
8. https://hackmd.io/@lido2370/S1aX6e1nN?type=view
9. https://pypi.org/project/tqdm/

## Other

- MacOS 遇到 `ModuleNotFoundError: No module named '_tkinter'`

	```shell
	brew install python-tk
	```

- 更新 pip

    ```shell
    python -m pip install --upgrade pip
    ```
