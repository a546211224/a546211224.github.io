import uvicorn
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

animal_classification = FastAPI()

# 加载模型
model = tf.keras.models.load_model("C:\\Users\\73197\\Desktop\\model")

# 动物类别标签定义
animal_labels = {
    0: '狗',
    1: '马',
    2: '大象',
    3: '蝴蝶',
    4: '鸡',
    5: '猫',
    6: '牛'
}

# 图片预处理函数
def preprocess_image(image):
    image = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_COLOR)
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(RGB_img, (224, 224))
    img_resize = preprocess_input(img_resize)
    img_reshape = img_resize[np.newaxis, ...]
    return img_reshape

# 定义 animal_classification 属性

# 定义预测端点
@animal_classification.post("/predict")
async def predict(upload_file: UploadFile = File(...)):
    contents = await upload_file.read()
    img_reshape = preprocess_image(contents)
    prediction = model.predict(img_reshape)
    max_pred_position = np.argmax(prediction)
    animal = animal_labels[max_pred_position]
    return {"animal_category": animal}

# 启动服务器
if __name__ == "__main__":
    uvicorn.run(animal_classification, host="0.0.0.0", port=8001)


