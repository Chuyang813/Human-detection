# 人物检测模型

本项目包含一个使用 Keras 框架构建的卷积神经网络 (CNN) 模型，用于实时检测摄像头捕获的视频流中是否有人物。

## 安装

首先，确保已经安装了以下库：

- OpenCV
- Tensorflow
- Keras

如果尚未安装，可以使用 pip 进行安装：

```bash
pip install opencv-python tensorflow keras
```

此外，您还需要安装一个适合您的操作系统的 [Python](https://www.python.org/) 版本。

## 使用

在开始之前，请确保你已经有一个预训练好的模型文件。如果你没有，你可以使用 `train.py` 脚本来训练模型。训练数据应该放在 `data` 文件夹下，其中 `1` 代表有人的图像，`0` 代表没有人的图像。

运行 `model.py` 脚本来训练模型。训练结束后，模型会被保存为 `model.h5`。

```bash
python model.py
```

之后，你可以使用 `test.py` 来运行人物检测。

```bash
python test.py
```

该脚本会打开默认摄像头并在新的窗口中显示摄像头捕获的视频流。预测的结果会以文本的形式显示在每一帧上。

按 'q' 键退出。

## 注意事项

本项目的模型是一个二元分类模型，而非对象检测模型。它并不能准确的定位视频中人的位置，只能预测视频中是否有人。

此外，由于模型的性能受限于训练数据的质量和数量，其预测的准确性可能会受到影响。在训练模型时，尽可能提供多样化并且充足的训练数据。

## 免责声明

本项目仅供学习和研究使用，不得用于任何非法用途。使用者对使用本项目产生的任何后果负全部责任。
