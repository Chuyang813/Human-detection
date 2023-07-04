# Human Detection Model

This project contains a convolutional neural network (CNN) model built using the Keras framework, which is used for real-time human detection in video streams captured by a camera.

## Installation

Firstly, make sure you have the following libraries installed:

- OpenCV
- Tensorflow
- Keras

If you haven't installed them, you can install them using pip:

```bash
pip install opencv-python tensorflow keras
```

In addition, you need to install a version of [Python](https://www.python.org/) suitable for your operating system.

## Usage

Before you start, make sure you have a pre-trained model file. If you don't, you can use the `train.py` script to train the model. The training data should be placed under the `data` folder, where `1` represents images with humans and `0` represents images without humans.

Run the `model.py` script to train the model. After the training is over, the model will be saved as `model.h5`.

```bash
python model.py
```

Then, you can use `test.py` to run the human detection.

```bash
python test.py
```

The script will open the default camera and display the video stream captured by the camera in a new window. The prediction results will be displayed as text on each frame.

Press 'q' to quit.

## Note

The model of this project is a binary classification model, not an object detection model. It cannot accurately locate humans in the video but can predict whether there are humans in the video.

Moreover, the performance of the model is limited by the quality and quantity of the training data, so its prediction accuracy may be affected. Try to provide diversified and sufficient training data when training the model.

## Disclaimer

This project is for learning and research purposes only and must not be used for any illegal purposes. The user is fully responsible for any consequences resulting from the use of this project.

