import streamlit as st
import requests
import cv2
import numpy as np

st.title("Animal Classifier")

upload_file = st.file_uploader("Please choose an image...", type=['jpg', 'png'])

animal_labels = {
    0: '狗',
    1: '马',
    2: '大象',
    3: '蝴蝶',
    4: '鸡',
    5: '猫',
    6: '牛'
}

if upload_file is not None:
    image = np.array(bytearray(upload_file.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(RGB_img, (224, 224))

    st.markdown("### Uploaded Image:")
    st.image(RGB_img, channels="RGB")

    img_reshape = img_resize[np.newaxis, ...]

    st.markdown("**Click the button to predict**")
    predict = st.button("Predict Category")
    if predict:
        # Send image to the backend for prediction
        # Replace 'http://localhost:8000/predict/' with your actual backend endpoint
        url = 'http://localhost:8001/predict/'  # Replace with your actual backend endpoint
        _, img_encoded = cv2.imencode('.png', RGB_img)
        response = requests.post(url, files={"upload_file": img_encoded.tobytes()})
        data_received = response.json()
        animal_category = data_received['animal_category']
        st.title(f"Predicted animal category: {animal_category}")
