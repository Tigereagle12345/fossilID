import tensorflow as tf
from PIL import Image
import numpy as np
import os

activation = "sigmoid"
im_path = "D:/Eamon/Documents/Coding/Python\ScienceFair/testImages/megalodon/images.jpg"
im_dir = "D:/Eamon/Documents/Coding/Python/ScienceFair/testImages/megalodon"
im_class = "megalodon"

model = tf.keras.models.load_model(f'{activation}_model.h5', compile = True)

def predict(im_path, im_dir, im_class):

    img_width, img_height = 180, 180
    img = tf.keras.preprocessing.image.load_img(im_path, target_size=(img_width, img_height))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    prediction = model.predict(img)

    predicted_class = np.argmax(prediction, axis=1)

    # Copied from output of train_ds.classnames in main.py
    dirs = ['Hemipristis', 'aduncus', 'alabamensis', 'auriculatus', 'chubutensis', 'contortus', 'cuvier', 'hastalis', 'mayumbensis', 'megalodon', 'obliquus', 'praecursor', 'taurus']

    percentages = tf.nn.softmax(prediction[0]).numpy() * 100

    print(prediction)
    print(["{:.20f}%".format(p) for p in percentages])
    print(f"Predicted class: {dirs[int(predicted_class)]}\nCorrect class: {im_class}")
    if im_class == dirs[int(predicted_class)]:
        print("Prediction Correct!")
    else:
        print("Prediction Incorrect!")

predict(im_path, im_dir, im_class)
