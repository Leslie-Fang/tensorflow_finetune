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
    label = raw_image.split("/")[-1].split(".")[0]
    return label

def read_train():
    res = tf.gfile.Glob("../dataset/test/*.jpg")
    images = np.asarray(list(map(preprocess, res)))
    ids = np.asarray(list(map(preprocess2, res)))
    print(images.shape)
    print(ids.shape)
    return images, ids

def test_fp32(batchsize):
	base_path = os.getcwd()
	# Get the Input
	test_data, ids = read_train()
	results = []
	with tf.Session() as sess:
		#restore from pb模型
		from tensorflow.python.platform import gfile
		with gfile.FastGFile(os.path.join(base_path,"pb_models")+'/test.pb', 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			for node in graph_def.node:
				print("node name is: {} \t node op is: {}".format(node.name,node.op))
			sess.graph.as_default()
			tf.import_graph_def(graph_def, name='') # 导入计算图
			Ys = sess.graph.get_tensor_by_name('Ys:0')
			X = sess.graph.get_tensor_by_name('input:0')
			iterations = int(test_data.shape[0] / batchsize)

			for step in range(iterations):
				print("step:{}".format(step))
				start = step * batchsize
				end = (step + 1) * batchsize
				inference_x = test_data[start:end, :]
				res = sess.run(Ys, feed_dict={X: inference_x})
				for item in res:
					#print(item)
					results.append(np.argmax(item))
		print("Model Type is FP32")
		print("Test Batchsize is :{}".format(batchsize))
		ids = np.asarray(ids).astype(np.int32)
		results = np.asarray(results).astype(np.int32)
		final = np.stack((ids,results),axis=-1)
		final = final[final[:,0].argsort()]
		with open("./sample_submission.csv", 'w') as f:
			spamwriter = csv.writer(f)
			spamwriter.writerow(['id', 'label'])
			for i in range(0, iterations * batchsize):
				spamwriter.writerow([final[i][0], final[i][1]])

def test(mode):
	if mode == 0:
		#FP32
		test_fp32(500)




