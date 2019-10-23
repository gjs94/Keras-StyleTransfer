# coding: utf-8
from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
from scipy.misc import imsave
from keras.applications import vgg19
from keras import backend as K
import os
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageEnhance, ImageFilter
 
# 输入参数
parser = argparse.ArgumentParser(description='基于Keras的图像风格迁移.')  # 解析器
parser.add_argument('--style_reference_image_path', metavar='ref', type=str,default = './style.jpg',
                    help='目标风格图片的位置')
parser.add_argument('--base_image_path', metavar='ref', type=str,default = './base.jpg',
                    help='基准图片的位置')
parser.add_argument('--iter', type=int, default=25, required=False,
                    help='迭代次数')
parser.add_argument('--pictrue_size', type=int, default=500, required=False,
                    help='图片大小.')
 
# 获取参数
args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
iterations = args.iter
pictrue_size = args.pictrue_size
 
 
source_image = Image.open(base_image_path)
source_image= source_image.resize((pictrue_size, pictrue_size))
 
width, height = pictrue_size, pictrue_size
 
 
def save_img(fname, image, image_enhance=True):  # 图像增强
    image = Image.fromarray(image)
    if image_enhance:
        # 亮度增强
        enh_bri = ImageEnhance.Brightness(image)
        brightness = 1.2
        image = enh_bri.enhance(brightness)
 
        # 色度增强
        enh_col = ImageEnhance.Color(image)
        color = 1.2
        image = enh_col.enhance(color)
 
        # 锐度增强
        enh_sha = ImageEnhance.Sharpness(image)
        sharpness = 1.2
        image = enh_sha.enhance(sharpness)
    imsave(fname, image)
    return
 
 
# util function to resize and format pictures into appropriate tensors
def preprocess_image(image):
    """
    预处理图片，包括变形到(1，width, height)形状，数据归一到0-1之间
    :param image: 输入一张图片
    :return: 预处理好的图片
    """
    image = image.resize((width, height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # (width, height)->(1，width, height)
    image = vgg19.preprocess_input(image)  # 0-255 -> 0-1.0
    return image
 
def deprocess_image(x):
    """
    将0-1之间的数据变成图片的形式返回
    :param x: 数据在0-1之间的矩阵
    :return: 图片，数据都在0-255之间
    """
    x = x.reshape((width, height, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')  # 以防溢出255范围
    return x
 
 
def gram_matrix(x):  # Gram矩阵
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram
 
# 风格损失，是风格图片与结果图片的Gram矩阵之差，并对所有元素求和
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    S_C = S-C
    channels = 3
    size = height * width
    return K.sum(K.square(S_C)) / (4. * (channels ** 2) * (size ** 2))
    #return K.sum(K.pow(S_C,4)) / (4. * (channels ** 2) * (size ** 2))  # 居然和平方没有什么不同
    #return K.sum(K.pow(S_C,4)+K.pow(S_C,2)) / (4. * (channels ** 2) * (size ** 2))  # 也能用，花后面出现了叶子
 
 
def eval_loss_and_grads(x):  # 输入x，输出对应于x的梯度和loss
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, height, width))
    else:
        x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])  # 输入x，得到输出
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values
 
# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(base, combination):
    return K.sum(K.square(combination - base))
 
# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x,img_nrows=width, img_ncols=height):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))
 
 
# Evaluator可以只需要进行一次计算就能得到所有的梯度和loss
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
 
    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
 
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
 
 
# 得到需要处理的数据，处理为keras的变量（tensor），处理为一个(3, width, height, 3)的矩阵
# 分别是基准图片，风格图片，结果图片
base_image = K.variable(preprocess_image(source_image))   # 基准图像
style_reference_image = K.variable(preprocess_image(load_img(style_reference_image_path)))
if K.image_data_format() == 'channels_first':
    combination_image = K.placeholder((1, 3, width, height))
else:
    combination_image = K.placeholder((1, width, height, 3))
 
# 组合以上3张图片，作为一个keras输入向量
input_tensor = K.concatenate([base_image, style_reference_image, combination_image], axis=0)   #组合
 
# 使用Keras提供的训练好的Vgg19网络,不带3个全连接层
model = vgg19.VGG19(input_tensor=input_tensor,weights='imagenet', include_top=False)
model.summary()  # 打印出模型概况
'''
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, None, None, 3)     0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, None, None, 64)    1792             A
_________________________________________________________________
block1_conv2 (Conv2D)        (None, None, None, 64)    36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, None, None, 64)    0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, None, None, 128)   73856            B
_________________________________________________________________
block2_conv2 (Conv2D)        (None, None, None, 128)   147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, None, None, 128)   0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, None, None, 256)   295168           C
_________________________________________________________________
block3_conv2 (Conv2D)        (None, None, None, 256)   590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, None, None, 256)   590080
_________________________________________________________________
block3_conv4 (Conv2D)        (None, None, None, 256)   590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, None, None, 256)   0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, None, None, 512)   1180160          D
_________________________________________________________________
block4_conv2 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block4_conv4 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, None, None, 512)   0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, None, None, 512)   2359808          E
_________________________________________________________________
block5_conv2 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block5_conv4 (Conv2D)        (None, None, None, 512)   2359808          F
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, None, None, 512)   0
=================================================================
'''
# Vgg19网络中的不同的名字，储存起来以备使用
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
 
loss = K.variable(0.)
 
layer_features = outputs_dict['block5_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
content_weight = 0.08
loss += content_weight * content_loss(base_image_features,
                                      combination_features)
 
feature_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
feature_layers_w = [0.1,0.1,0.4,0.3,0.1]
# feature_layers = ['block5_conv1']
# feature_layers_w = [1]
for i in range(len(feature_layers)):
    # 每一层的权重以及数据
    layer_name, w = feature_layers[i], feature_layers_w[i]
    layer_features = outputs_dict[layer_name]  # 该层的特征
 
    style_reference_features = layer_features[1, :, :, :]  # 参考图像在VGG网络中第i层的特征
    combination_features = layer_features[2, :, :, :]     # 结果图像在VGG网络中第i层的特征
 
    loss += w * style_loss(style_reference_features, combination_features)  # 目标风格图像的特征和结果图像特征之间的差异作为loss
 
loss += total_variation_loss(combination_image)
 
 
# 求得梯度，输入combination_image，对loss求梯度, 每轮迭代中combination_image会根据梯度方向做调整
grads = K.gradients(loss, combination_image)
 
outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)
 
f_outputs = K.function([combination_image], outputs)
 
evaluator = Evaluator()
x = preprocess_image(source_image)
img = deprocess_image(x.copy())
fname = '原始图片.png'
save_img(fname, img)
 
# 开始迭代
for i in range(iterations):
    start_time = time.time()
    print('迭代', i,end="   ")
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20, epsilon=1e-7)
    # 一个scipy的L-BFGS优化器
    print('目前loss:', min_val,end="  ")
    # 保存生成的图片
    img = deprocess_image(x.copy())
 
    fname = 'result_%d.png' % i
    end_time = time.time()
    print('耗时%.2f s' % (end_time - start_time))
 
    if i%5 == 0 or i == iterations-1:
        save_img(fname, img, image_enhance=True)
        print('文件保存为', fname)