#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

import tensorflow as tf

def tf_multi_hot(x, depth):
  def func_c(p, v):
    return tf.less(p, tf.shape(x)[0])

  def func_b(p, v):
    row = tf.add_n(tf.unstack(indexes[p]))
    return p + 1, tf.concat([v, [row]], axis=0)

  indexes = tf.one_hot(x, depth)
  initV = tf.constant(0)

  _, v = tf.while_loop(func_c, func_b,
                        [initV, tf.convert_to_tensor([list(range(depth))],
                                                    tf.float32)],
                        shape_invariants=[initV.get_shape(),
                                          tf.TensorShape([None, depth])])

  return v[1:,]


def log_sum(tensor_list):
  '''
  :param tensor_list: All tensors are of [batch, 1] shape.
  '''
  tensor = tf.concat(tensor_list, 1)
  return tf.reduce_logsumexp(tensor, 1)

def create_bi_LSTM(word_ids,
                   vob_size,
                   embedding_size,
                   LSTM_layer_num):
  def encode(input, reuse):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
      embeddings = tf.get_variable("embeddings", [vob_size, embedding_size])
      cell = rnn_cell.MultiRNNCell([rnn_cell.LSTMCell(embedding_size)
                                    for _ in range(LSTM_layer_num)])
      word_vec = tf.nn.embedding_lookup(embeddings, input)
      word_list = tf.unstack(word_vec, axis=1)
      outputs, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)
      return outputs
  
  rnn_cell = tf.nn.rnn_cell
  outputs1 = encode(word_ids, reuse=False)
  
  rev_words_ids = [reversed(list(sample)) for sample in word_ids]
  outputs2 = encode(rev_words_ids, reuse=True)
  outputs2 = list(reversed(outputs2))
  
  outputs = [tf.concat(o, axis=1) for o in zip(outputs1, outputs2)]
  return outputs
