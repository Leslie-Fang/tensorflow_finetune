# -*- coding: utf-8 -*-
import pandas as pd
import csv
import struct
from PIL import Image
import tensorflow as tf
import numpy as np
import datetime
import os

def read_train():
	data = pd.read_csv("../dataset/train.csv")
	data = data.values #pandas dataframe 转换成 numpy.ndarray 格式
	data[:,1:785] = data[:,1:785]/255.0
	return data

def train():
	log_step = 1000
	learning_rate = 0.0001
	batchsize = 64
	epoch = 10
	print("Begin train!")
	starttime = datetime.datetime.now()

	raw_data = read_train()

	input_image_size = raw_data.shape[1]-1
	# print(input_image_size)
	X = tf.placeholder(tf.float32,[None,input_image_size],name="X")
	x_image = tf.reshape(X,[-1,28,28,1]) #转换成矩阵之后可以进行卷积运算，reshape API https://blog.csdn.net/m0_37592397/article/details/78695318 -1表示由计算过程自动去指定
	Y_ = tf.placeholder(tf.float32,[None,10],name="Y_") #10表示手写数字识别的10个类别
	#conv1
	layer1_weights = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.05))
	layer1_bias = tf.Variable(tf.constant(0.1,shape=[32]))
	layer1_conv = tf.nn.conv2d(x_image,layer1_weights,strides=[1,1,1,1],padding='SAME')#https://www.cnblogs.com/lizheng114/p/7498328.html
	layer1_relu = tf.nn.relu(layer1_conv+layer1_bias)#https://blog.csdn.net/m0_37870649/article/details/80963053
	layer1_pool = tf.nn.max_pool(layer1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #https://blog.csdn.net/mzpmzk/article/details/78636184
	#conv2
	layer2_weights = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.05))
	layer2_bias = tf.Variable(tf.constant(0.1,shape=[64]))
	layer2_conv = tf.nn.conv2d(layer1_pool,layer2_weights,strides=[1,1,1,1],padding='SAME')
	layer2_relu = tf.nn.relu(layer2_conv+layer2_bias)
	layer2_pool = tf.nn.max_pool(layer2_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	#FC
	layer3_weights = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.05))
	layer3_bias = tf.Variable(tf.constant(0.1,shape=[1024]))
	layer3_flat = tf.reshape(layer2_pool,[-1,7*7*64])#展开成一维，进行全连接层的计算
	layer3_relu = tf.nn.relu(tf.matmul(layer3_flat,layer3_weights)+layer3_bias)
	#Dropout_layer
	keep_prob = tf.placeholder(tf.float32,name="keep_prob")
	h_fc1_drop = tf.nn.dropout(layer3_relu,keep_prob)
	#FC2
	layer4_weights = tf.Variable(tf.truncated_normal([1024,10],stddev=0.05))
	layer4_bias = tf.Variable(tf.constant(0.1,shape=[10]))
	#Ys = tf.nn.softmax(tf.matmul(layer3_relu,layer4_weights)+layer4_bias,name="Ys")
	Ys = tf.nn.softmax(tf.matmul(h_fc1_drop,layer4_weights)+layer4_bias,name="Ys")  # The output is like [0 0 1 0 0 0 0 0 0 0]
	y_pred_cls = tf.argmax(Ys,dimension=1)
	loss = -tf.reduce_mean(Y_*tf.log(Ys))
	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
	count = 0
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for e in range(epoch):
			#每个epoch开始的时候，打乱顺序
			np.random.shuffle(raw_data) #会打乱传入数组的元素的顺序
			train_label = raw_data[:,0]
			train_data = raw_data[:,1:785]

			iterations = int(train_data.shape[0]/batchsize)
			print("iterations in one epoch is: {}".format(iterations))
			for step in range(iterations):

				start = step * batchsize
				end = (step+1) * batchsize
				label_vals = train_label[start:end]
				train_image_pixs = train_data[start:end,:]

				train_y_label = []
				for item in label_vals:
					train_sub_y_label = []
					for i in range(10):
						if item != i:
							train_sub_y_label.append(0)
						else:
							train_sub_y_label.append(1)
					train_y_label.append(train_sub_y_label)
				train_x = np.array(train_image_pixs,dtype=np.float32)
				train_y = np.array(train_y_label,dtype=np.float32)

				sess.run(train_op,feed_dict={X:train_x, Y_:train_y,keep_prob:0.4})
				if (int(step) % int(log_step)) == 0:
					c = sess.run(loss,feed_dict={X:train_x, Y_:train_y,keep_prob:0.4})
					print("epoch:{0}, Step:{1}, loss:{2}".format(e+1,step,c))

		print("Running configuration learning_rate:{}, batchsize:{}, epoch:{}".format(learning_rate,batchsize,epoch))
		endtime = datetime.datetime.now()
		print("The program takes:{} sec".format((endtime - starttime).seconds))
		base_path = os.getcwd()
		#save ckpt 格式的模型
		if os.path.isdir(os.path.join(base_path,"checkPoint")) is False:
			os.makedirs(os.path.join(base_path,"checkPoint"))
		saver = tf.train.Saver()
		saver.save(sess,os.path.join(base_path,"checkPoint/trainModel"))
		#save pb文件
		if os.path.isdir(os.path.join(base_path,"pb_models")) is False:
			os.makedirs(os.path.join(base_path,"pb_models"))
		from tensorflow.python.framework import graph_util
		#最后一个参数用于指定输出节点的名字
		#把所有变量转换成常量，并保留他们的值
		constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Ys'])
		with tf.gfile.FastGFile(os.path.join(base_path,"pb_models")+'/model.pb', mode='wb') as f:
			f.write(constant_graph.SerializeToString())
