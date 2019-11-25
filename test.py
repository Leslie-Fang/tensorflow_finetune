# -*- coding: utf-8 -*-
import pandas as pd
import csv
import tensorflow as tf
import numpy as np
import datetime
import os
from tensorflow.keras.preprocessing import image

def preprocess(raw_image):
    img = image.load_img(raw_image, target_size=(224, 224))  # match the input of exsiting models
    img = image.img_to_array(img) / 255.0
    return img

def preprocess2(raw_image):
    id = raw_image.split("/")[-1].split(".")[0]
    return id

def read_train():
    res = tf.compat.v1.io.gfile.glob("../dataset/test/*.jpg")
    images = np.asarray(list(map(preprocess, res)))
    ids = np.asarray(list(map(preprocess2, res)))
    print(images.shape)
    print(ids.shape)
    return images, ids

def test_fp32():
    base_path = os.getcwd()
    # Get the Input
    test_data, ids = read_train()
    results = []
    new_model = tf.compat.v1.keras.experimental.load_from_saved_model('pb_models')
    new_model.summary()
    res = new_model.predict(test_data)
    for item in res:
        results.append(np.argmax(item))
    ids = np.asarray(ids).astype(np.int32)
    results = np.asarray(results).astype(np.int32)
    final = np.stack((ids, results), axis=-1)
    final = final[final[:, 0].argsort()]
    with open("./sample_submission.csv", 'w') as f:
        spamwriter = csv.writer(f)
        spamwriter.writerow(['id', 'label'])
        for i in range(0, test_data.shape[0]):
            spamwriter.writerow([final[i][0], final[i][1]])
