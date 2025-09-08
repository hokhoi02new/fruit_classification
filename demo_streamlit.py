
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import streamlit as st
import tempfile


model_path = 'model/model.h5'

def predict_class_img_with_img(img_arr):
    class_names = ['Apple', 'Banana', 'Carambola', 'Guava', 'Kiwi', 'Mango', 'Orange', 'Peach', 'Pear', 'Persimmon', 
                   'Pitaya', 'Plum', 'Pomegranate', 'Tomatoes', 'muskmelon']

    # load model
    model = load_model(model_path)

    np_img = np.array(img_arr)

    # print(np_img.shape)
    image = np.expand_dims(cv2.resize(np.squeeze(np_img), (160, 160)), axis=0)

    predictions = model.predict(image)
    scores = tf.nn.sigmoid(predictions)
    pred_labels = np.argmax(scores, axis=-1)
    return class_names[int(pred_labels)]


if __name__ == '__main__':
    st.title("Chương trình phân loại trái cây: ")
    f = st.file_uploader("Upload file")
    print(f)
    if f:
        if f.name[-3:] in ('jpg', 'png', 'JPG', 'PNG'):
            st.image(f)
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(f.read())
        vf = cv2.VideoCapture(tfile.name)
        _, frame = vf.read()
        st.write("Wating...")
        Class_name = predict_class_img_with_img(frame)
        st.write("Kết quả phân loại trái cây: ")
        st.header(Class_name)