from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LambdaCallback
from sklearn.model_selection import train_test_split
import os

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 设定图片大小和训练参数
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 10
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)  # 3 表示 RGB颜色通道

# 设定图片的存放路径


print("build model...")

# 创建一个卷积神经网络模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # 二分类，所以用sigmoid激活函数
])

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

data_dir = r'C:\Users\lcy08\Desktop\1\Human-detection\data'
# 使用ImageDataGenerator来读取和预处理图片
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  # 使用20%的数据作为验证集

print('load data...')

# 训练数据生成器
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training')  # set as training data

# 验证数据生成器
validation_generator = train_datagen.flow_from_directory(
    data_dir, 
    target_size=(IMG_HEIGHT, IMG_WIDTH), 
    batch_size=BATCH_SIZE, 
    class_mode='binary',
    subset='validation')  # set as validation data

print('train model...')

simple_log = LambdaCallback(on_epoch_end = lambda e, l: 
    print(f'Epoch {e+1}: loss={l["loss"]}, accuracy={l["accuracy"]}, val_loss={l["val_loss"]}, val_accuracy={l["val_accuracy"]}'))

# 训练模型
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS)

model.save('model.h5')
