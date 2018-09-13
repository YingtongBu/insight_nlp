#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

import tensorflow as tf

def accuracy(prediction, label, name):
  correct = tf.equal(prediction, label)
  return tf.reduce_mean(tf.cast(correct, "float"), name=name)

def lookup0(table, pos):
  '''
  :param table: [width1, width2]
  :param pos: [batch] or [batch, seq_length]
  :return: [batch, width2] or [batch, seq_length, width2]
  '''
  return tf.nn.embedding_lookup(table, pos)

def lookup1(table, table_width, pos):
  '''
  :param table: [batch, table_width]
  :param pos:   [batch]
  :return [batch]
  '''
  dtype = table.dtype
  return tf.reduce_sum(tf.multiply(table,
                                   tf.one_hot(pos, table_width, dtype=dtype)),
                       axis=1)
  
def lookup2(table, pos):
  '''
  :param table: [table_width]
  :param pos:   [batch]
  :return [batch]
  '''
  return tf.nn.embedding_lookup(table, pos)
  
def lookup3(table, table_width, pos1, pos2):
  '''
  :param table: [table_width, table_width]
  :param pos1: [batch]
  :param pos2: [batch]
  :return [batch]
  '''
  col = lookup0(table, pos1)
  return lookup1(col, table_width, pos2)

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
  :param tensor_list: All tensors are of [batch] or [batch, 1] shape.
  '''
  assert len(tensor_list) > 0
  tensor_list = [tf.reshape(t, [tf.size(t), 1]) for t in tensor_list]
  tensor = tf.concat(tensor_list, 1)
  return tf.reduce_logsumexp(tensor, 1)

def create_bi_LSTM(word_ids, vob_size, embedding_size, LSTM_layer_num,
                   RNN_type="lstm"):
  '''
  :param word_ids: tensor, of shape [batch_size, length]
  :param vob_size:
  :param embedding_size:
  :param LSTM_layer_num:
  :return:
  '''
  def encode(input, score_name, reuse):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
      embeddings = tf.get_variable("embeddings", [vob_size, embedding_size])
     
    with tf.variable_scope(score_name, reuse=False):
      if RNN_type.lower() == "lstm":
        cell = rnn_cell.LSTMCell
      elif RNN_type.lower() == "gru":
        cell = rnn_cell.GRUCell
      else:
        assert False
        
      cell = rnn_cell.MultiRNNCell([cell(embedding_size)
                                    for _ in range(LSTM_layer_num)])
      word_vec = lookup0(embeddings, input)
      word_list = tf.unstack(word_vec, axis=1)
      outputs, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)
      return outputs
  
  rnn_cell = tf.nn.rnn_cell
  outputs1 = encode(word_ids, "directed", reuse=False)
  outputs2 = encode(tf.reverse(word_ids, [1]), "reversed", reuse=True)
  outputs2 = list(reversed(outputs2))
  
  outputs = [tf.concat(o, axis=1) for o in zip(outputs1, outputs2)]
  return outputs
