#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

import tensorflow as tf

class TF:
  @staticmethod
  def multiHot(x, depth):
    def funcC(p, v):
      return tf.less(p, tf.shape(x)[0])

    def funcB(p, v):
      row = tf.add_n(tf.unstack(indexes[p]))
      return p + 1, tf.concat([v, [row]], axis=0)

    indexes = tf.one_hot(x, depth)
    initV = tf.constant(0)

    _, v = tf.while_loop(funcC, funcB,
                         [initV, tf.convert_to_tensor([list(range(depth))],
                                                      tf.float32)],
                         shape_invariants=[initV.get_shape(),
                                           tf.TensorShape([None, depth])])

    return v[1:,]

