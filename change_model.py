# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
import pandas as pd
import numpy as np
import os

def print_model(model_path):
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(f.read())
        for node in graph_def.node:
            print("node name is: {} \t node op is: {}".format(node.name, node.op))

def read_train():
    data = pd.read_csv("../dataset/train.csv")
    data = data.values #pandas dataframe 转换成 numpy.ndarray 格式
    data[:,1:785] = data[:,1:785]/255.0
    return data

def change_model(model_path):
    input_nodes_dict = {}
    # Import the origin model
    with gfile.FastGFile(model_path, 'rb') as f:
        input_graph = graph_pb2.GraphDef()
        input_graph.ParseFromString(f.read())
        for node in input_graph.node:
            input_nodes_dict[node.name] = node
    # Extract the subgraph, refer to the train.py and  netron's model name
    from tensorflow.python.framework import graph_util
    output_graph = graph_util.extract_sub_graph(input_graph, ["dropout/mul"])

    # Define fine-tune parameters
    log_step = 1000
    learning_rate = 0.0001
    batchsize = 64
    epoch = 10

    # Get the input dataset
    raw_data = read_train()
    with tf.Session() as sess:
        # 导入计算图
        sess.graph.as_default()
        tf.import_graph_def(output_graph, name='')
        Ys_origin = sess.graph.get_tensor_by_name('dropout/mul:0')
        X = sess.graph.get_tensor_by_name('X:0')
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')

        # Add new node and layers
        Y_ = tf.placeholder(tf.float32, [None, 10], name="Y_")  # 10表示手写数字识别的10个类别
        layer4_weights = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.05))
        layer4_bias = tf.Variable(tf.constant(0.1, shape=[10]))
        Ys = tf.nn.softmax(tf.matmul(Ys_origin, layer4_weights) + layer4_bias, name="Ys")  # The output is like [0 0 1 0 0 0 0 0 0 0]
        loss = -tf.reduce_mean(Y_ * tf.log(Ys))
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # Fine-tune
        sess.run(tf.global_variables_initializer())
        for e in range(epoch):
            # 每个epoch开始的时候，打乱顺序
            np.random.shuffle(raw_data)  # 会打乱传入数组的元素的顺序
            train_label = raw_data[:, 0]
            train_data = raw_data[:, 1:785]

            iterations = int(train_data.shape[0] / batchsize)
            print("iterations in one epoch is: {}".format(iterations))
            for step in range(iterations):

                start = step * batchsize
                end = (step + 1) * batchsize
                label_vals = train_label[start:end]
                train_image_pixs = train_data[start:end, :]

                train_y_label = []
                for item in label_vals:
                    train_sub_y_label = []
                    for i in range(10):
                        if item != i:
                            train_sub_y_label.append(0)
                        else:
                            train_sub_y_label.append(1)
                    train_y_label.append(train_sub_y_label)
                train_x = np.array(train_image_pixs, dtype=np.float32)
                train_y = np.array(train_y_label, dtype=np.float32)

                sess.run(train_op, feed_dict={X: train_x, Y_: train_y, keep_prob: 0.4})
                if (int(step) % int(log_step)) == 0:
                    c = sess.run(loss, feed_dict={X: train_x, Y_: train_y, keep_prob: 0.4})
                    print("epoch:{0}, Step:{1}, loss:{2}".format(e, step, c))

    #
        print("Running configuration learning_rate:{}, batchsize:{}, epoch:{}".format(learning_rate, batchsize, epoch))
        base_path = os.getcwd()
        # save ckpt 格式的模型
        if os.path.isdir(os.path.join(base_path, "checkPoint")) is False:
            os.makedirs(os.path.join(base_path, "checkPoint"))
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(base_path, "checkPoint/trainModel"))
        # save pb文件
        if os.path.isdir(os.path.join(base_path, "pb_models")) is False:
            os.makedirs(os.path.join(base_path, "pb_models"))
        # 最后一个参数用于指定输出节点的名字
        # 把所有变量转换成常量，并保留他们的值
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Ys'])
        with tf.gfile.FastGFile(os.path.join(base_path, "pb_models") + '/test.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

if __name__ == "__main__":
    change_model("./pb_models/model.pb")
    # print_model("./pb_models/test.pb")