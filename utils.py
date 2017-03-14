from __future__ import division
import scipy.misc
import numpy as np
import tensorflow as tf


def get_image(image_path, img_size, is_grayscale=False):
  image = imread(image_path, is_grayscale)
  return transform(image, img_size)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
  if (is_grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

#merge all sample_num sample pictures into one picture
def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  return img

def imsave(images, size, path):
  return scipy.misc.imsave(path, merge(images, size))

def transform(image, img_size):
  resized_image = scipy.misc.imresize(image, [img_size, img_size])
  return np.array(resized_image)/127.5 - 1.

def inverse_transform(images):
  return (images+1)*127.5

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def conv2d(input_, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def transpose_conv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.2,
       name="transpose_conv2d"):
  """
  0.2 = sqrt(1/(5*5)) = sqrt(Var(w)) = sqrt(1/(k_h*k_w))
  在卷积的计算中，假设输入为4*4，卷积核大小为2*2，stride为2，那么输出为2*2(padding=valid)
  首先将输入reshape为1*16，输出reshape为1*4，那么y=xc,这里c为16*4的稀疏权值矩阵，在转置卷积中
  我们要对输入进行上采样，即从y还原x，那么y*c(t)=x，c(t)为c的转置矩阵，但是从y=xc直观来看，应该有
  y*c(-1)=x, c(-1)位c的逆矩阵, 那么c(-1)=c(t)所以c(t)*c=I，I为单位矩阵，从c(t)*c=I可得
  Σwi*wi = 1 (0 <= i <= k_h*k_w)    (1)
  Σwi*wj = 0 (0 <= i <= k_h*k_w, 0 <= j <= k_h*k_w, i != j)    (2)
  假设每个w都是独立同分布的，那么从(1)可得
  E(w*w) = 1/(k_h*k_w)
  从(2)可得
  E(wi*wj) = E(wi)*E(wj) = 0
  即E(w) = 0，所以
  E(w*w)=Var(w)=1/(k_h*k_w)
  """
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    tconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
              strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    tconv = tf.reshape(tf.nn.bias_add(tconv, biases), tconv.get_shape())

    return tconv
     
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))

    return tf.matmul(input_, matrix) + bias
