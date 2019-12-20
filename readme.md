## Introduction
利用预训练的resnet50模型进行迁移学习 fine-tune 解决新的问题

## Dataset download
Kaggle playground cat_or_dog

## Note
### categorical_crossentropy
多分类问题，二分类问题也可以，target格式就是[0, 1]
最后一层用softmax去激活

### binary_crossentropy
二分类问题，结果应该是1个值，在0-1之间，target的格式是一个数字
最后一层用sigmoid去激活

### resnet50 和 imagenet
输入的图片不需要归一到0-1之间
