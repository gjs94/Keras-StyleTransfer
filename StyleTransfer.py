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
 
# �������
parser = argparse.ArgumentParser(description='����Keras��ͼ����Ǩ��.')  # ������
parser.add_argument('--style_reference_image_path', metavar='ref', type=str,default = './style.jpg',
                    help='Ŀ����ͼƬ��λ��')
parser.add_argument('--base_image_path', metavar='ref', type=str,default = './base.jpg',
                    help='��׼ͼƬ��λ��')
parser.add_argument('--iter', type=int, default=25, required=False,
                    help='��������')
parser.add_argument('--pictrue_size', type=int, default=500, required=False,
                    help='ͼƬ��С.')
 
# ��ȡ����
args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
iterations = args.iter
pictrue_size = args.pictrue_size
 
 
source_image = Image.open(base_image_path)
source_image= source_image.resize((pictrue_size, pictrue_size))
 
width, height = pictrue_size, pictrue_size
 
 
def save_img(fname, image, image_enhance=True):  # ͼ����ǿ
    image = Image.fromarray(image)
    if image_enhance:
        # ������ǿ
        enh_bri = ImageEnhance.Brightness(image)
        brightness = 1.2
        image = enh_bri.enhance(brightness)
 
        # ɫ����ǿ
        enh_col = ImageEnhance.Color(image)
        color = 1.2
        image = enh_col.enhance(color)
 
        # �����ǿ
        enh_sha = ImageEnhance.Sharpness(image)
        sharpness = 1.2
        image = enh_sha.enhance(sharpness)
    imsave(fname, image)
    return
 
 
# util function to resize and format pictures into appropriate tensors
def preprocess_image(image):
    """
    Ԥ����ͼƬ���������ε�(1��width, height)��״�����ݹ�һ��0-1֮��
    :param image: ����һ��ͼƬ
    :return: Ԥ����õ�ͼƬ
    """
    image = image.resize((width, height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # (width, height)->(1��width, height)
    image = vgg19.preprocess_input(image)  # 0-255 -> 0-1.0
    return image
 
def deprocess_image(x):
    """
    ��0-1֮������ݱ��ͼƬ����ʽ����
    :param x: ������0-1֮��ľ���
    :return: ͼƬ�����ݶ���0-255֮��
    """
    x = x.reshape((width, height, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')  # �Է����255��Χ
    return x
 
 
def gram_matrix(x):  # Gram����
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram
 
# �����ʧ���Ƿ��ͼƬ����ͼƬ��Gram����֮���������Ԫ�����
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    S_C = S-C
    channels = 3
    size = height * width
    return K.sum(K.square(S_C)) / (4. * (channels ** 2) * (size ** 2))
    #return K.sum(K.pow(S_C,4)) / (4. * (channels ** 2) * (size ** 2))  # ��Ȼ��ƽ��û��ʲô��ͬ
    #return K.sum(K.pow(S_C,4)+K.pow(S_C,2)) / (4. * (channels ** 2) * (size ** 2))  # Ҳ���ã������������Ҷ��
 
 
def eval_loss_and_grads(x):  # ����x�������Ӧ��x���ݶȺ�loss
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, height, width))
    else:
        x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])  # ����x���õ����
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
 
 
# Evaluator����ֻ��Ҫ����һ�μ�����ܵõ����е��ݶȺ�loss
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
 
 
# �õ���Ҫ��������ݣ�����Ϊkeras�ı�����tensor��������Ϊһ��(3, width, height, 3)�ľ���
# �ֱ��ǻ�׼ͼƬ�����ͼƬ�����ͼƬ
base_image = K.variable(preprocess_image(source_image))   # ��׼ͼ��
style_reference_image = K.variable(preprocess_image(load_img(style_reference_image_path)))
if K.image_data_format() == 'channels_first':
    combination_image = K.placeholder((1, 3, width, height))
else:
    combination_image = K.placeholder((1, width, height, 3))
 
# �������3��ͼƬ����Ϊһ��keras��������
input_tensor = K.concatenate([base_image, style_reference_image, combination_image], axis=0)   #���
 
# ʹ��Keras�ṩ��ѵ���õ�Vgg19����,����3��ȫ���Ӳ�
model = vgg19.VGG19(input_tensor=input_tensor,weights='imagenet', include_top=False)
model.summary()  # ��ӡ��ģ�͸ſ�
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
# Vgg19�����еĲ�ͬ�����֣����������Ա�ʹ��
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
    # ÿһ���Ȩ���Լ�����
    layer_name, w = feature_layers[i], feature_layers_w[i]
    layer_features = outputs_dict[layer_name]  # �ò������
 
    style_reference_features = layer_features[1, :, :, :]  # �ο�ͼ����VGG�����е�i�������
    combination_features = layer_features[2, :, :, :]     # ���ͼ����VGG�����е�i�������
 
    loss += w * style_loss(style_reference_features, combination_features)  # Ŀ����ͼ��������ͽ��ͼ������֮��Ĳ�����Ϊloss
 
loss += total_variation_loss(combination_image)
 
 
# ����ݶȣ�����combination_image����loss���ݶ�, ÿ�ֵ�����combination_image������ݶȷ���������
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
fname = 'ԭʼͼƬ.png'
save_img(fname, img)
 
# ��ʼ����
for i in range(iterations):
    start_time = time.time()
    print('����', i,end="   ")
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20, epsilon=1e-7)
    # һ��scipy��L-BFGS�Ż���
    print('Ŀǰloss:', min_val,end="  ")
    # �������ɵ�ͼƬ
    img = deprocess_image(x.copy())
 
    fname = 'result_%d.png' % i
    end_time = time.time()
    print('��ʱ%.2f s' % (end_time - start_time))
 
    if i%5 == 0 or i == iterations-1:
        save_img(fname, img, image_enhance=True)
        print('�ļ�����Ϊ', fname)