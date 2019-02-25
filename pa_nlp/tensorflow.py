#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from pa_nlp import *
from pa_nlp.common import print_flush
import tensorflow as tf

xavier_init   = tf.contrib.layers.xavier_initializer
norm_init     = tf.truncated_normal_initializer
rand_init     = tf.random_uniform_initializer
const_init    = tf.constant_initializer

def matmul(m1: tf.Tensor, m2: tf.Tensor)-> tf.Tensor:
  shape1 = m1.shape.as_list()
  shape2 = m2.shape.as_list()
  if shape1[: -2] == shape2[: -2] and shape1[-1] == shape2[-2]:
    return m1 @ m2

  if shape1[-1] == shape2[0] and len(shape2) == 2:
    m1 = tf.reshape(m1, [-1, shape1[-1]])
    m = m1 @ m2
    shape = shape1[: -1] + [shape2[1]]
    m = tf.reshape(m, shape)
    return m
  
  assert False

def tf_bytes_feature(value: bytes):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def tf_float_feature(value: float):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def tf_int64_feature(value: int):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_tfrecord(samples: typing.Union[list, typing.Iterator],
                   serialize_sample_fun, file_name: str):
  with tf.python_io.TFRecordWriter(file_name) as writer:
    num = 0
    for sample in samples:
      for example in serialize_sample_fun(sample):
        num += 1
        if num % 1000 == 0:
          print_flush(f"{num} examples have been finished.")
        writer.write(example)

def read_tfrecord(file_name: str,
                  example_fmt: dict, example2sample_func,
                  epoch_num: int, batch_size: int):
  def parse_fn(example):
    parsed = tf.parse_single_example(example, example_fmt)
    return example2sample_func(parsed)

  def input_fn():
    files = tf.data.Dataset.list_files(file_name)
    dataset = files.apply(
      tf.data.experimental.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=4)
    )

    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(epoch_num)
    dataset = dataset.apply(
      tf.data.experimental.map_and_batch(
        map_func=parse_fn, batch_size=batch_size,
      )
    )

    return dataset

  dataset = input_fn()
  data_iter = dataset.prefetch(8).make_initializable_iterator()
  sample = data_iter.get_next()

  return data_iter.initializer, sample

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

def highway_layer(input, size, num_layers=1, activation=tf.nn.relu,
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
  return tf.reduce_sum(
    tf.multiply(table, tf.one_hot(pos, table_width, dtype=dtype)),
    axis=1
  )
  
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

def multi_hot(x, depth):
  def func_c(p, v):
    return tf.less(p, tf.shape(x)[0])

  def func_b(p, v):
    row = tf.add_n(tf.unstack(indexes[p]))
    return p + 1, tf.concat([v, [row]], axis=0)

  indexes = tf.one_hot(x, depth)
  initV = tf.constant(0)

  _, v = tf.while_loop(
    func_c,
    func_b,
    [initV, tf.convert_to_tensor([list(range(depth))], tf.float32)],
    shape_invariants=[initV.get_shape(), tf.TensorShape([None, depth])]
  )

  return v[1:,]

def log_sum(tensor_list: list):
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

def basic_attention(states: list, context: tf.Tensor)-> tf.Tensor:
  '''
  status: list of [batch, hidden-unit]
  context = tf.Variable(
    tf.random_uniform([hidden_unit], -1., 1), dtype=tf.float32
  )
  '''
  status = tf.stack(states)
  status = tf.transpose(status, [1, 0, 2])

  scores = tf.reduce_sum(status * context, 2)
  probs = tf.nn.softmax(scores)
  probs = tf.expand_dims(probs, 2)

  return tf.reduce_sum(status * probs, 1)

def attention_global(state: tf.Tensor, name: str)-> tf.Tensor:
  '''
  global attention.
  status: [batch, max-time, hidden-unit]
  context = tf.Variable(
    tf.random_uniform([hidden_unit], -1., 1), dtype=tf.float32
  )
  '''
  with tf.name_scope(name):
    h = tf.get_variable(
      name, (state.shape[2], 1), tf.float32, rand_init(-1, 1)
    )

  scores = matmul(state, h)
  probs = tf.nn.softmax(scores, axis=1)

  return tf.reduce_sum(state * probs, 1)

def attention_basic1(state: tf.Tensor, context: tf.Tensor,
                     name: str)-> tf.Tensor:
  '''
  state: [batch, max-time, hidden-unit]
  context: [batch, hidden-unit]
  <x, y> = x * H * y
  '''
  shape = state.shape
  max_time, h_size = shape[1], shape[2]

  state = tf.transpose(state, [1, 0, 2])  # [max-time, batch, hidden-unit]
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    h = tf.get_variable(
      name, (h_size, h_size), tf.float32, rand_init(-1, 1)
    )

  scores = tf.reduce_sum(matmul(state, h) * context, 2)
  probs = tf.nn.softmax(scores, axis=0)
  probs = tf.expand_dims(probs, -1)
  vec = tf.reduce_sum(state * probs, 0)

  return vec

def attention_self1(state: tf.Tensor, name: str)-> tf.Tensor:
  '''
  :param state: [batch, max-time, hidden-unit]
  '''
  max_time, h_size = state.shape[1:]
  results = []
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    for time_step in range(max_time):
      context = state[:, time_step, :]
      vec = attention_basic1(state, context, f"element")
      results.append(vec)

  result = tf.stack(results)
  result = tf.transpose(result, [1, 0, 2])

  return result

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

def normalize_data(data: tf.Tensor,
                   axels: list,
                   method: str="mean"    # mean, or guassian
                   )-> tf.Tensor:
  shape = data.shape
  trans1 = axels[:]
  for p in range(len(shape)):
    if p not in trans1:
      trans1.append(p)

  data1 = tf.transpose(data, trans1)
  mean_ts, var_ts = tf.nn.moments(data1, list(range(len(axels))))

  method = method.lower()
  if method == "mean":
    data2 = data1 - mean_ts
  elif method == "guassian":
    data2 = (data1 - mean_ts) / tf.sqrt(var_ts + 1e-8)
  else:
    assert False

  trans2 = list(range(len(trans1)))
  for p in trans1:
    trans2[trans1[p]] = p

  data3 = tf.transpose(data2, trans2)

  return data3


