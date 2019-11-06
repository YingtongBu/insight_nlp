#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

from pa_nlp import *
from pa_nlp.nlp import print_flush
import tensorflow as tf

def to_double(tensor, type=tf.float32):
  return tf.cast(tensor, type)

def to_int(tensor, type=tf.int32):
  return tf.cast(tensor, type)



