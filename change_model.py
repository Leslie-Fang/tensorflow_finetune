# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import graph_util
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing import image

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
    #img = np.expand_dims(img, 0)
    return img

def preprocess2(raw_image):
    label = raw_image.split("/")[-1].split(".")[0]
    # print(raw_image)
    # print(label)
    if label == "dog":
        label = [0, 1]
    else:
        label = [1, 0]
    # print(label)
    return label

def read_train():
    res = tf.gfile.Glob("../dataset/train/*.*.jpg")
    np.random.shuffle(res)
    #images = None
    images = np.asarray(list(map(preprocess, res)))
    labels = np.asarray(list(map(preprocess2, res)))
    print(images.shape)
    print(labels.shape)
    return images, labels

def change_model(model_path):
    #Get the Input
    train_data, train_label = read_train()

    input_nodes_dict = {}
    # Import the origin model
    with gfile.FastGFile(model_path, 'rb') as f:
        input_graph = graph_pb2.GraphDef()
        input_graph.ParseFromString(f.read())
        for node in input_graph.node:
            input_nodes_dict[node.name] = node
    # Extract the subgraph, refer to the train.py and  netron's model name
    output_graph = graph_util.extract_sub_graph(input_graph, ["v0/affine0/add"])

    # Define fine-tune parameters
    log_step = 10
    learning_rate = 0.001
    batchsize = 100
    epoch = 10

    config = tf.ConfigProto(intra_op_parallelism_threads=26,
                            inter_op_parallelism_threads=2,
                            allow_soft_placement=True,
                            device_count={'CPU': 26})

    with tf.Session(config=config) as sess:
        # 导入计算图
        sess.graph.as_default()
        tf.import_graph_def(output_graph, name='')
        Ys_origin = sess.graph.get_tensor_by_name('v0/affine0/add:0')
        X = sess.graph.get_tensor_by_name('input:0')

        # Add new node and layers
        Y_ = tf.placeholder(tf.float32, [None, 2], name="Y_")  # 2: cat or dog
        layer4_weights = tf.Variable(tf.truncated_normal([1001, 2], stddev=0.05))
        layer4_bias = tf.Variable(tf.constant(0.1, shape=[2]))
        Ys = tf.nn.softmax(tf.matmul(Ys_origin, layer4_weights) + layer4_bias, name="Ys")  # The output is like [0 0 1 0 0 0 0 0 0 0]
        loss = -tf.reduce_mean(Y_ * tf.log(Ys))
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # Fine-tune
        sess.run(tf.global_variables_initializer())
        for e in range(epoch):
            iterations = int(train_label.shape[0] / batchsize)
            for step in range(iterations):

                start = step * batchsize
                end = (step + 1) * batchsize
                train_y_label = train_label[start:end]
                train_image_pixs = train_data[start:end, :]

                train_x = np.array(train_image_pixs, dtype=np.float32)
                train_y = np.array(train_y_label, dtype=np.float32)

                sess.run(train_op, feed_dict={X: train_x, Y_: train_y})
                if (int(step) % int(log_step)) == 0:
                    c = sess.run(loss, feed_dict={X: train_x, Y_: train_y})
                    print("epoch:{0}, Step:{1}, loss:{2}".format(e, step, c))

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
    change_model("./pb_models/freezed_resnet50.pb")
    # print_model("./pb_models/test.pb")