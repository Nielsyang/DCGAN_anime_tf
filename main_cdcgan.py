import os
import scipy.misc
import numpy as np

from dcgan_use_condition import DCGAN

import tensorflow as tf
from tensorflow.python import debug as tf_debug


flags = tf.app.flags
flags.DEFINE_integer("epoch", 30, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for d network")
flags.DEFINE_float("beta1", 0.5, "Momentum of MomentumOptimizer")
flags.DEFINE_integer("train_size", 480, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("img_size", 64, "image size")
flags.DEFINE_integer("condition", 1, "limited conditions to test")
flags.DEFINE_integer("channel_dim", 3, "Dimension of image color")
flags.DEFINE_string("dataset", "anime_use_cond", "The name of dataset")
flags.DEFINE_string("filename_pattern", "*.jpg", "Glob pattern of filename of input images")
flags.DEFINE_string("checkpoint_dir", "checkpoint_use_cond", "Directory name to save the checkpoints")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples")
flags.DEFINE_string("test_dir", "tests", "Directory name to save the test image samples")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  # sess = tf.Session()
  # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
  with tf.Session() as sess:
    dcgan = DCGAN(
        sess,
        img_size=FLAGS.img_size,
        batch_size=FLAGS.batch_size,
        channel_dim=FLAGS.channel_dim,
        dataset_name=FLAGS.dataset,
        filename_pattern=FLAGS.filename_pattern,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir)

    if FLAGS.is_train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir):
        raise Exception("No availabel checkpoint file")
      else:
        dcgan.test(FLAGS)

if __name__ == '__main__':
  tf.app.run()
