from model import IMG_HEIGHT, IMG_WIDTH
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('model.h5')

def process_frame(frame):
    # 将捕获的图像调整为模型输入的大小，并增加一个维度使其成为批次的一部分
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))  # 需要替换成模型的输入尺寸
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    
    # 使用模型进行预测
    prediction = model.predict(frame)
    
    return prediction

cap = cv2.VideoCapture(0)

while True:
    # 从摄像头读取帧
    ret, frame = cap.read()
    if not ret:
        break
    
    # 使用模型处理帧并获取预测结果
    prediction = process_frame(frame)
    
    # 根据预测结果在图像上添加文字
    if prediction > 0.5:
        text = 'People'
    else:
        text = 'No People'
    
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 显示带有预测结果的图像
    cv2.imshow('frame', frame)
    
    # 如果按下'q'键，停止循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 完成后，释放摄像头并销毁所有窗口
cap.release()
cv2.destroyAllWindows()