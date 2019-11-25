# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow import keras

def print_model(model_path):
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(f.read())
        for node in graph_def.node:
            print("node name is: {} \t node op is: {}".format(node.name, node.op))
#
def preprocess(raw_image):
    img = image.load_img(raw_image, target_size=(224, 224))  # match the input of exsiting models
    img = image.img_to_array(img) / 255.0
    return img

def preprocess2(raw_image):
    label = raw_image.split("/")[-1].split(".")[0]
    if label == "dog":
        label = [0, 1]
    else:
        label = [1, 0]
    return label

def read_train():
    res = tf.compat.v1.io.gfile.glob("../dataset/train/*.*.jpg")
    np.random.shuffle(res)
    images = np.asarray(list(map(preprocess, res)))
    labels = np.asarray(list(map(preprocess2, res)))
    print(images.shape)
    print(labels.shape)
    return images, labels

def train():
    NUM_CLASSES = 2
    model = Sequential()
    model.add(keras.applications.ResNet50(include_top=False, pooling='max', weights="imagenet"))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    # ResNet-50 model is already trained, should not be trained
    model.layers[0].trainable = True
    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    X_train, y_train = read_train()
    train_model = model.fit(X_train, y_train,
                            batch_size=512,
                            epochs=1,
                            verbose=1,
                            )
    tf.compat.v1.keras.experimental.export_saved_model(model, 'pb_models')
    print('export saved model.')
    del model  # 删除网络对象

if __name__ == "__main__":
    train()