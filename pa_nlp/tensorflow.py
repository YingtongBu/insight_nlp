#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

import tensorflow as tf
import typing
from operator import itemgetter

activations = tf.keras.activations
estimator = tf.estimator
layers = tf.layers
losses = tf.losses
nn = tf.nn
rnn_cell = tf.nn.rnn_cell

norm_init1 = tf.truncated_normal_initializer(stddev=0.1)
rand_init1 = tf.random_uniform_initializer(-1, 1)
const_init = tf.constant_initializer

def load_model(graph: tf.Graph, sess: tf.Session, model_path: str):
  try:
    with graph.as_default():
      tf.train.Saver().restore(sess, tf.train.latest_checkpoint(model_path))
    print(f"Successful loading existing model from '{model_path}'")
    return True

  except Exception as error:
    print(f"Failed loading existing model from '{model_path}: {error}")
    return False

def get_network_parameter_num():
  num = 0
  for var in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = var.get_shape()
    var_param_num = 1
    for dim in shape:
      var_param_num *= dim.value
    num += var_param_num
  print(f"#model parameter: {num}")

  return num

def construct_optimizer(
  loss: tf.Tensor,
  learning_rate: typing.Union[float, tf.Tensor]=0.001,
  gradient_norm: typing.Union[float, None]=None
):
  batch_id = tf.train.create_global_step()
  opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    gradient = opt.compute_gradients(loss=loss)
    if gradient_norm is None:
      opt_op = opt.apply_gradients(gradient, global_step=batch_id)

    else:
      grads = list(map(itemgetter(0), gradient))
      parms = list(map(itemgetter(1), gradient))
      grads, _ = tf.clip_by_global_norm(grads, gradient_norm)
      opt_op = opt.apply_gradients(list(zip(grads, parms)),
                                   global_step=batch_id)

    return opt_op

def high_way_layer(input, size, num_layers=1, activation=tf.nn.relu,
                   scope='highway'):
  '''
   t = sigmoid(Wy + b)
   z = t * g(Wy + b) + (1 - t) * y
   where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
 '''
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    output = input
    for idx in range(num_layers):
      prob = tf.sigmoid(tf.layers.dense(input, size))
      g = activation(tf.layers.dense(input, size))

      output = prob * g + (1. - prob) * input
      input = output
  
    return output

def accuracy(prediction, label):
  correct = tf.equal(prediction, label)
  return tf.reduce_mean(tf.cast(correct, "float"))

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

def create_bi_LSTM(
  word_vec: tf.Tensor,  # [batch, max_len, embedding_size]
  rnn_layer_num: int,
  hidden_unit_num: int,
  rnn_type: str= "lstm"
)-> list:
  def encode(input, score_name):
    with tf.variable_scope(score_name, reuse=False):
      if rnn_type.lower() == "lstm":
        cell = rnn_cell.LSTMCell
      elif rnn_type.lower() == "gru":
        cell = rnn_cell.GRUCell
      else:
        assert False
        
      cell = rnn_cell.MultiRNNCell(
        [cell(hidden_unit_num) for _ in range(rnn_layer_num)]
      )
      word_list = tf.unstack(input, axis=1)
      outputs, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

      return outputs

  rnn_cell = tf.nn.rnn_cell
  outputs1 = encode(word_vec, "directed")
  outputs2 = encode(tf.reverse(word_vec, [1]), "reversed")
  outputs2 = list(reversed(outputs2))
  
  outputs = [tf.concat(o, axis=1) for o in zip(outputs1, outputs2)]

  return outputs

def basic_attention(statuses: list, context: tf.Tensor)-> tf.Tensor:
  '''
  status: list of [batch, hidden-unit]
  context = tf.Variable(
    tf.random_uniform([hidden_unit], -1., 1), dtype=tf.float32
  )
  '''
  status = tf.stack(statuses)
  status = tf.transpose(status, [1, 0, 2])

  scores = tf.reduce_sum(status * context, 2)
  probs = tf.nn.softmax(scores)
  probs = tf.expand_dims(probs, 2)

  return tf.reduce_sum(status * probs, 1)

def batch_norm_wrapper(inputs, scope_name, is_train: bool, decay=0.99,
                       float_type=tf.float32):
  epsilon = 1e-3
  shape = inputs.get_shape().as_list()

  with tf.variable_scope(f"bn_{scope_name}"):
    offset = tf.get_variable("offset", shape[-1], dtype=float_type,
                             initializer=const_init(0))
    scale = tf.get_variable("scale", shape[-1], dtype=float_type,
                            initializer=const_init(1))
    pop_mean = tf.get_variable("mean", shape[-1], dtype=float_type,
                               initializer=const_init(0))
    pop_var = tf.get_variable("variance", shape[-1], dtype=float_type,
                              initializer=const_init(1))
    if is_train:
      batch_mean, batch_var = tf.nn.moments(
        inputs, axes=list(range(len(shape)-1))
      )
      update_pop_mean = tf.assign(
        pop_mean, pop_mean * decay + batch_mean * (1 - decay)
      )
      update_pop_var =tf.assign(
        pop_var, pop_var * decay + batch_var * (1 - decay)
      )

      with tf.control_dependencies([update_pop_mean, update_pop_var]):
        return tf.nn.batch_normalization(
          inputs, batch_mean, batch_var, offset, scale, epsilon
        )

    else:
      return tf.nn.batch_normalization(
        inputs, pop_mean, pop_var, offset, scale, epsilon
      )

