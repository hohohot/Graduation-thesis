import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import functools
from PIL import Image

"""
TensorFlow Data Augmentation Example
Reference : https://github.com/tensorflow/models/blob/master/research/object_detection/core/preprocessor.py
Author : solaris33
Project URL : http://solarisailab.com/archives/2619
"""

"""
shape augmentation function list:
  1. random_horizontal_flip(X)
  2. random_vertical_flip(X)
  3. random_rotation90(X)
  4. random_image_scale(X)
  5. random_pad_image(X)
  6. random_crop_to_aspect_ratio(X)
  7. random_pad_to_aspect_ratio(X)
color augmentation function list:
  1. random_pixel_value_scale(X),
  2. random_rgb_to_gray(X)
  3. random_adjust_brightness(X)
  4. random_adjust_contrast(X)
  5. random_adjust_hue(X),
  6. random_adjust_saturation(X)
  7. random_distort_color(X)
miscellaneous augmentation function list:
  1. random_black_patches(X)
"""

def _random_integer(minval, maxval, seed):
  """Returns a random 0-D tensor between minval and maxval.
  Args:
    minval: minimum value of the random tensor.
    maxval: maximum value of the random tensor.
    seed: random seed.
  Returns:
    A random 0-D tensor between minval and maxval.
  """
  return tf.random.uniform(
      [], minval=minval, maxval=maxval, dtype=tf.int32, seed=seed)

def random_adjust_brightness(image,
                             max_delta=0.2,
                             seed=None):
    delta = tf.random.uniform([], -max_delta, max_delta, seed=seed)
    image = tf.image.adjust_brightness(image / 255, delta) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)

    return image

def random_adjust_contrast(image,
                           min_delta=0.8,
                           max_delta=1.25,
                           seed=None):
    contrast_factor = tf.random.uniform([], min_delta, max_delta, seed=seed)
    image = tf.image.adjust_contrast(image / 255, contrast_factor) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)

    return image

def random_adjust_hue(image,
                      max_delta=0.02,
                      seed=None):
    delta = tf.random.uniform([], -max_delta, max_delta, seed=seed)
    image = tf.image.adjust_hue(image / 255, delta) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)

    return image

def random_adjust_saturation(image,
                             min_delta=0.8,
                             max_delta=1.25,
                             seed=None):
    saturation_factor = tf.random.uniform([], min_delta, max_delta, seed=seed)
    image = tf.image.adjust_saturation(image / 255, saturation_factor) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)

    return image

def random_distort_color(image, color_ordering=0):
    if color_ordering == 0:
      image = random_adjust_brightness(
          image, max_delta=32. / 255.)
      image = random_adjust_saturation(
          image)
      image = random_adjust_hue(
          image, max_delta=0.08)
      image = random_adjust_contrast(
          image)
      image = tf.clip_by_value(image+tf.random.normal(mean=0, stddev=10, shape=image.shape, dtype=tf.dtypes.double), 0, 255)
      

    elif color_ordering == 1:
      image = random_adjust_brightness(
          image, max_delta=32. / 255.)
      image = random_adjust_contrast(
          image, min_delta=0.5, max_delta=1.5)
      image = random_adjust_saturation(
          image, min_delta=0.5, max_delta=1.5)
      image = random_adjust_hue(
          image, max_delta=0.2)
    else:
      raise ValueError('color_ordering must be in {0, 1}')
    return image




def data_agument(x, postAguModel):
    x = random_distort_color(x, 0)
    x = postAguModel(x)
    return x
    
    
    