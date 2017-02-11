from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from utils import *

class DCGAN(object):
  def __init__(self, sess, img_size=64, batch_size=64, sample_num = 64,
         fake_data_dim=100, g_filter_num=64, d_filter_num=64, channel_dim=3, 
         dataset_name='default', filename_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):

    self.sess = sess
    self.is_grayscale = (channel_dim == 1)

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.img_size = img_size

    self.fake_data_dim = fake_data_dim

    self.g_filter_num = g_filter_num
    self.d_filter_num = d_filter_num

    self.channel_dim = channel_dim

    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.filename_pattern = filename_pattern
    self.checkpoint_dir = checkpoint_dir
    self.build_model()

  def build_model(self):
    image_dims = [self.img_size, self.img_size, self.channel_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    #if shape = None you can feed any shape of tensor 
    self.fake = tf.placeholder(
      tf.float32, [None, self.fake_data_dim], name='fake_data')

    self.G = self.generator(self.fake)
    self.D_real, self.D_real_logits = self.discriminator(inputs)

    self.sampler = self.sampler(self.fake)
    self.D_fake, self.D_fake_logits = self.discriminator(self.G, reuse=True)

    self.d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_real_logits, targets=tf.ones_like(self.D_real)))

    self.d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_fake_logits, targets=tf.zeros_like(self.D_fake)))

    self.g_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_fake_logits, targets=tf.ones_like(self.D_fake)))

    self.prob_d_real = tf.reduce_mean(self.D_real)
    self.prob_d_fake = tf.reduce_mean(self.D_fake)

    self.prob_d_real_sum = tf.scalar_summary("prob_d_real", self.prob_d_real)
    self.prob_d_fake_sum = tf.scalar_summary("prob_d_fake", self.prob_d_fake)
    self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    data = glob(os.path.join("./data", config.dataset, self.filename_pattern))

    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = tf.merge_summary([self.d_loss_fake_sum, self.g_loss_sum, self.prob_d_fake_sum])
    self.d_sum = tf.merge_summary([self.d_loss_real_sum, self.d_loss_sum, self.prob_d_real_sum])
    self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

    sample_fake = np.random.uniform(-1, 1, size=(self.sample_num , self.fake_data_dim))
    
    sample_files = data[0:self.sample_num]
    sample = [get_image(sample_file, img_size=self.img_size, is_grayscale=self.is_grayscale) for sample_file in sample_files]
    if (self.is_grayscale):
      sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
      sample_inputs = np.array(sample).astype(np.float32)
  
    counter = 1
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print("Load model success")
    else:
      print("Load model failed")

    data = glob(os.path.join("./data", config.dataset, self.filename_pattern))
    for epoch in xrange(config.epoch):      
      batch_idxs = min(len(data), config.train_size) // config.batch_size
      np.random.shuffle(data)

      for idx in xrange(0, batch_idxs):
        batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
        batch = [get_image(batch_file, img_size=self.img_size, is_grayscale=self.is_grayscale) for batch_file in batch_files]
        if (self.is_grayscale):
          batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
        else:
          batch_images = np.array(batch).astype(np.float32)

        batch_fake = np.random.uniform(-1, 1, [config.batch_size, self.fake_data_dim]).astype(np.float32)

        _, summary_str = self.sess.run([d_optim, self.d_sum],
                         feed_dict={ self.inputs:batch_images, self.fake:batch_fake })
        self.writer.add_summary(summary_str, counter)

        _, summary_str = self.sess.run([g_optim, self.g_sum],
                         feed_dict={ self.fake:batch_fake })
        self.writer.add_summary(summary_str, counter) 

        # Run g_optim twice because batch size of D network is real images(64) + fake images(64)
        # but G's batch size only fake images(64)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
                         feed_dict={ self.fake:batch_fake })
        self.writer.add_summary(summary_str, counter)
        

        errD_fake, errD_real, errG, prob_d_fake, prob_d_real = self.sess.run(
        	[self.d_loss_fake, self.d_loss_real, self.g_loss, self.prob_d_fake, self.prob_d_real],
        	feed_dict={self.fake:batch_fake, self.inputs:batch_images})

        counter += 1
        print("Epoch:[%2d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, d_fake_prob: %.5f, d_real_prob: %.5f" \
          % (epoch, time.time()-start_time, errD_fake+errD_real, errG, prob_d_fake, prob_d_real))

        if np.mod(counter, batch_idxs) == 0:
          try:
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.fake:sample_fake,
                  self.inputs:sample_inputs,
              },
            )
            save_images(samples, [8, 8],
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
          except:
            print("Save pic error!")

        if np.mod(counter, batch_idxs*2) == 0:
          self.save(config.checkpoint_dir, counter)

  def discriminator(self, image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.d_filter_num, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.d_filter_num*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.d_filter_num*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.d_filter_num*8, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

      #sigmoid change h4 to prob because h4 may not in [0,1]
      #but h4 is the parameter because sigmoid_cross_entroy_with_logits loss function
      #include a sigmoid function
      return tf.nn.sigmoid(h4), h4

  #all trainable weight and bias are defined in ops(conv2d,trans_conv2d,linear)
  def generator(self, fake_data):
    with tf.variable_scope("generator") as scope:
      s_h, s_w = self.img_size, self.img_size
      s_h2, s_h4, s_h8, s_h16 = \
          int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
      s_w2, s_w4, s_w8, s_w16 = \
          int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

      fake_= linear(
          fake_data, self.g_filter_num*8*s_h16*s_w16, 'g_h0_lin')

      h0 = tf.reshape(
          fake_, [-1, s_h16, s_w16, self.g_filter_num * 8])
      h0 = tf.nn.relu(self.g_bn0(h0))

      h1= transpose_conv2d(
          h0, [self.batch_size, s_h8, s_w8, self.g_filter_num*4], name='g_h1')
      h1 = tf.nn.relu(self.g_bn1(h1))

      h2= transpose_conv2d(
          h1, [self.batch_size, s_h4, s_w4, self.g_filter_num*2], name='g_h2')
      h2 = tf.nn.relu(self.g_bn2(h2))

      h3= transpose_conv2d(
          h2, [self.batch_size, s_h2, s_w2, self.g_filter_num*1], name='g_h3')
      h3 = tf.nn.relu(self.g_bn3(h3))

      h4= transpose_conv2d(
          h3, [self.batch_size, s_h, s_w, self.channel_dim], name='g_h4')

      #tanh function normlized the output to [-1,1]
      #because we normlized the real image into [-1,1]
      #G's output is also D's input
      return tf.nn.tanh(h4)

  #sampler function sample fake images per 100 steps
  #and merge all fake images into one image for visualize
  #so sampler dont update the weight and bias parameters
  def sampler(self, fake_data):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()
        
      s_h, s_w = self.img_size, self.img_size
      s_h2, s_h4, s_h8, s_h16 = \
          int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
      s_w2, s_w4, s_w8, s_w16 = \
          int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

      h0 = tf.reshape(
          linear(fake_data, self.g_filter_num*8*s_h16*s_w16, 'g_h0_lin'),
          [-1, s_h16, s_w16, self.g_filter_num * 8])
      h0 = tf.nn.relu(self.g_bn0(h0, train=False))

      h1 = transpose_conv2d(h0, [self.batch_size, s_h8, s_w8, self.g_filter_num*4], name='g_h1')
      h1 = tf.nn.relu(self.g_bn1(h1, train=False))

      h2 = transpose_conv2d(h1, [self.batch_size, s_h4, s_w4, self.g_filter_num*2], name='g_h2')
      h2 = tf.nn.relu(self.g_bn2(h2, train=False))

      h3 = transpose_conv2d(h2, [self.batch_size, s_h2, s_w2, self.g_filter_num*1], name='g_h3')
      h3 = tf.nn.relu(self.g_bn3(h3, train=False))

      h4 = transpose_conv2d(h3, [self.batch_size, s_h, s_w, self.channel_dim], name='g_h4')

      return tf.nn.tanh(h4)

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    print("Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print("Success to read {}".format(ckpt_name))
      return True
    else:
      print("Failed to find a checkpoint")
      return False
