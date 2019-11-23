# -*- coding: utf-8 -*-
import pandas as pd
import csv
import tensorflow as tf
import numpy as np
import datetime
import os

def read_test():
	data = pd.read_csv("../dataset/test.csv")
	data = data.values/255.0
	print(data.shape)
	return data

def test_fp32(batchsize):
	print("Begin inference!")
	base_path = os.getcwd()

	raw_data = read_test()
	input_image_size = raw_data.shape[1]
	results = []
	with tf.Session() as sess:
		#restore from pb模型
		from tensorflow.python.platform import gfile
		with gfile.FastGFile(os.path.join(base_path,"pb_models")+'/model.pb', 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			for node in graph_def.node:
				print("node name is: {} \t node op is: {}".format(node.name,node.op))
			sess.graph.as_default()
			tf.import_graph_def(graph_def, name='') # 导入计算图
		#restore from ckpt格式的模型
		# saver = tf.train.import_meta_graph(os.path.join(base_path,"checkPoint/trainModel.meta"))
		# saver.restore(sess, tf.train.latest_checkpoint(os.path.join(base_path,"checkPoint")))
		iterations = int(raw_data.shape[0]/batchsize)
		totalTime = 0
		for step in range(iterations):
			start = step * batchsize
			end = (step+1) * batchsize
			inference_image_pixs = raw_data[start:end,:]
			inference_x = np.array(inference_image_pixs,dtype=np.float32)
			# 获取需要进行计算的operator
			Ys = sess.graph.get_tensor_by_name('Ys:0')
			X = sess.graph.get_tensor_by_name('X:0')
			keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')

			starttime = datetime.datetime.now()
			res = sess.run(Ys,feed_dict={X:inference_x, keep_prob:0.4})

			endtime = datetime.datetime.now()
			print("Iteration {} takes {} microseconds \t Throughput: {} images/second".format(step+1,(endtime-starttime).microseconds,batchsize*1000*1000/(endtime-starttime).microseconds))
			totalTime += (endtime-starttime).microseconds
			results.extend(map(lambda x: np.argmax(x),res))
	print("Model Type is FP32")
	print("Test Batchsize is :{}".format(batchsize))
	print("Average throughput is: {} images/second".format(iterations*batchsize*1000*1000/totalTime))
	with open("./sample_submission.csv",'w') as f:
		spamwriter = csv.writer(f)
		spamwriter.writerow(['ImageId','Label'])
		for i in range(0,iterations*batchsize):
			spamwriter.writerow([i+1,results[i]])


def test_fp32_new_model(batchsize):
	print("Begin inference!")
	base_path = os.getcwd()

	raw_data = read_test()
	input_image_size = raw_data.shape[1]
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
		#restore from ckpt格式的模型
		# saver = tf.train.import_meta_graph(os.path.join(base_path,"checkPoint/trainModel.meta"))
		# saver.restore(sess, tf.train.latest_checkpoint(os.path.join(base_path,"checkPoint")))
		iterations = int(raw_data.shape[0]/batchsize)
		totalTime = 0
		for step in range(iterations):
			start = step * batchsize
			end = (step+1) * batchsize
			inference_image_pixs = raw_data[start:end,:]
			inference_x = np.array(inference_image_pixs,dtype=np.float32)
			# 获取需要进行计算的operator
			Ys = sess.graph.get_tensor_by_name('Ys:0')
			X = sess.graph.get_tensor_by_name('X:0')
			keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')

			starttime = datetime.datetime.now()
			res = sess.run(Ys,feed_dict={X:inference_x, keep_prob:0.4})

			endtime = datetime.datetime.now()
			print("Iteration {} takes {} microseconds \t Throughput: {} images/second".format(step+1,(endtime-starttime).microseconds,batchsize*1000*1000/(endtime-starttime).microseconds))
			totalTime += (endtime-starttime).microseconds
			results.extend(map(lambda x: np.argmax(x),res))
	print("Model Type is FP32")
	print("Test Batchsize is :{}".format(batchsize))
	print("Average throughput is: {} images/second".format(iterations*batchsize*1000*1000/totalTime))
	with open("./sample_submission.csv",'w') as f:
		spamwriter = csv.writer(f)
		spamwriter.writerow(['ImageId','Label'])
		for i in range(0,iterations*batchsize):
			spamwriter.writerow([i+1,results[i]])

def test(mode):
	if mode == 0:
		#FP32
		test_fp32(1)
	if mode == 1:
		#FP32 new model after fine tune
		test_fp32_new_model(1)



