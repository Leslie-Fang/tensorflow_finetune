## Dataset download
Kaggle playground cat_or_dog

## 运行步骤
TF1.14
* 运行change_mode.py 裁剪原始resnet50模型的最后1层，添加新的层，重新训练新的模型可以参考inference的代码，只要找到对应的op(adam) 然后session.run就可以了，保存新的模型成pb

## 修改模型基本思路change_mode.py
修改pb模型，可以参考INT8化时对模型修改的方法

先用netron看原来训练好的模型的结构和各个节点的名字

Tensorflow有python接口
graph_util提供了extract_sub_graph 等方法来修改图模型
https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/graph_util
参考这个代码：
https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/tools/quantization/quantize_graph.py
提供了很多对图的操作

## 局限性
目前train完之后，将模型的变量都转换成了常量去保存，因此这部分值不能再fine-tune了，参考train的代码：
```
#最后一个参数用于指定输出节点的名字
#把所有变量转换成常量，并保留他们的值
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Ys'])
```
https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/graph_util/convert_variables_to_constants
