#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

from __future__ import absolute_import
import keras
from keras import backend as K
from keras import regularizers
from keras import constraints
from keras import initializers
from keras.engine import Layer, InputSpec
import tensorflow as tf
'''
ChainCRF layer to do sequence labeling
'''
def path_energy(y, x, u, b_start=None, b_end=None, mask=None):
  x = add_boundary_energy(x, b_start, b_end, mask)
  return path_energy0(y, x, u, mask)

def path_energy0(y, x, u, mask=None):
  n_classes = K.shape(x)[2]
  y_one_hot = K.one_hot(y, n_classes)
  energy = K.sum(x * y_one_hot, 2)
  energy = K.sum(energy, 1)
  y_t = y[:, :-1]
  y_tp1 = y[:, 1:]
  u_flat = K.reshape(u, [-1])
  flat_indices = y_t * n_classes + y_tp1
  uyt_tp1 = K.gather(u_flat, flat_indices)
  if mask is not None:
    mask = K.cast(mask, K.floatx())
    y_t_mask = mask[:, :-1]
    y_tp1_mask = mask[:, 1:]
    uyt_tp1 *= y_t_mask * y_tp1_mask
  energy += K.sum(uyt_tp1, axis=1)
  return energy

def sparse_chain_crf_loss(y, x, u, b_start=None, b_end=None, mask=None):
  x = add_boundary_energy(x, b_start, b_end, mask)
  energy = path_energy0(y, x, u, mask)
  energy -= free_energy0(x, u, mask)
  return K.expand_dims(-energy, -1)

def chain_crf_loss(y, x, u, b_start=None, b_end=None, mask=None):
  y_sparse = K.argmax(y, -1)
  y_sparse = K.cast(y_sparse, 'int32')
  return sparse_chain_crf_loss(y_sparse, x, u, b_start, b_end, mask)

def add_boundary_energy(x, b_start=None, b_end=None, mask=None):
  if mask is None:
    if b_start is not None:
      x = K.concatenate([x[:, :1, :] + b_start, x[:, 1:, :]], axis=1)
    if b_end is not None:
      x = K.concatenate([x[:, :-1, :], x[:, -1:, :] + b_end], axis=1)
  else:
    mask = K.cast(mask, K.floatx())
    mask = K.expand_dims(mask, 2)
    x *= mask
    if b_start is not None:
      mask_r = K.concatenate([K.zeros_like(mask[:, :1]), mask[:, :-1]],
                            axis=1)
      start_mask = K.cast(K.greater(mask, mask_r), K.floatx())
      x = x + start_mask * b_start
    if b_end is not None:
      mask_l = K.concatenate([mask[:, 1:], K.zeros_like(mask[:, -1:])],
                            axis=1)
      end_mask = K.cast(K.greater(mask, mask_l), K.floatx())
      x = x + end_mask * b_end
  return x

def viterbi_decode(x, u, b_start=None, b_end=None, mask=None):
  x = add_boundary_energy(x, b_start, b_end, mask)

  alpha0 = x[:, 0, :]
  gamma0 = K.zeros_like(alpha0)
  initial_states = [gamma0, alpha0]
  _, gamma = _forward(x,
                      lambda B: [K.cast(K.argmax(B, axis=1),
                                 K.floatx()), K.max(B, axis=1)],
                      initial_states,
                      u,
                      mask)
  y = _backward(gamma, mask)
  return y

def free_energy(x, u, b_start=None, b_end=None, mask=None):
  x = add_boundary_energy(x, b_start, b_end, mask)
  return free_energy0(x, u, mask)

def free_energy0(x, u, mask=None):
  initial_states = [x[:, 0, :]]
  last_alpha, _ = _forward(x,
                          lambda B: [K.logsumexp(B, axis=1)],
                          initial_states,
                          u,
                          mask)
  return last_alpha[:, 0]

def _forward(x, reduce_step, initial_states, u, mask=None):
  def _forward_step(energy_matrix_t, states):
    alpha_tm1 = states[-1]
    new_states = reduce_step(K.expand_dims(alpha_tm1, 2) + energy_matrix_t)
    return new_states[0], new_states

  u_shared = K.expand_dims(K.expand_dims(u, 0), 0)
  if mask is not None:
    mask = K.cast(mask, K.floatx())
    mask_u = K.expand_dims(K.expand_dims(mask[:, :-1] * mask[:, 1:], 2), 3)
    u_shared = u_shared * mask_u
  inputs = K.expand_dims(x[:, 1:, :], 2) + u_shared
  inputs = K.concatenate([inputs, K.zeros_like(inputs[:, -1:, :, :])], axis=1)
  last, values, _ = K.rnn(_forward_step, inputs, initial_states)
  return last, values

def batch_gather(reference, indices):
  ref_shape = K.shape(reference)
  batch_size = ref_shape[0]
  n_classes = ref_shape[1]
  flat_indices = K.arange(0, batch_size) * n_classes + K.flatten(indices)
  return K.gather(K.flatten(reference), flat_indices)

def _backward(gamma, mask):
  gamma = K.cast(gamma, 'int32')

  def _backward_step(gamma_t, states):
    y_tm1 = K.squeeze(states[0], 0)
    y_t = batch_gather(gamma_t, y_tm1)
    return y_t, [K.expand_dims(y_t, 0)]

  initial_states = [K.expand_dims(K.zeros_like(gamma[:, 0, 0]), 0)]
  _, y_rev, _ = K.rnn(_backward_step,
                     gamma,
                     initial_states,
                     go_backwards=True)
  y = K.reverse(y_rev, 1)

  if mask is not None:
    mask = K.cast(mask, dtype='int32')
    y *= mask
    y += -(1 - mask)
  return y

class ChainCRF(Layer):
  def __init__(self, init='glorot_uniform',
               U_regularizer=None,
               b_start_regularizer=None,
               b_end_regularizer=None,
               U_constraint=None,
               b_start_constraint=None,
               b_end_constraint=None,
               weights=None,
               **kwargs):
    super(ChainCRF, self).__init__(**kwargs)
    self.init = initializers.get(init)
    self.U_regularizer = regularizers.get(U_regularizer)
    self.b_start_regularizer = regularizers.get(b_start_regularizer)
    self.b_end_regularizer = regularizers.get(b_end_regularizer)
    self.U_constraint = constraints.get(U_constraint)
    self.b_start_constraint = constraints.get(b_start_constraint)
    self.b_end_constraint = constraints.get(b_end_constraint)
    self.initial_weights = weights
    self.supports_masking = True
    self.uses_learning_phase = True
    self.input_spec = [InputSpec(ndim=3)]

  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) == 3
    return (input_shape[0], input_shape[1], input_shape[2])

  def compute_mask(self, input, mask=None):
    if mask is not None:
      return K.any(mask, axis=1)
    return mask

  def _fetch_mask(self):
    mask = None
    if self._inbound_nodes:
      mask = self._inbound_nodes[0].input_masks[0]

    return mask

  def build(self, input_shape):
    assert len(input_shape) == 3
    n_classes = input_shape[2]
    n_steps = input_shape[1]
    assert n_steps is None or n_steps >= 2
    self.input_spec = [InputSpec(dtype=K.floatx(),
                                 shape=(None, n_steps, n_classes))]

    self.u = self.add_weight((n_classes, n_classes),
                             initializer=self.init,
                             name='U',
                             regularizer=self.U_regularizer,
                             constraint=self.U_constraint)

    self.b_start = self.add_weight((n_classes, ),
                                   initializer='zero',
                                   name='b_start',
                                   regularizer=self.b_start_regularizer,
                                   constraint=self.b_start_constraint)

    self.b_end = self.add_weight((n_classes, ),
                                 initializer='zero',
                                 name='b_end',
                                 regularizer=self.b_end_regularizer,
                                 constraint=self.b_end_constraint) 

    if self.initial_weights is not None:
      self.set_weights(self.initial_weights)
      del self.initial_weights
   
    self.built = True

  def call(self, x, mask=None):
    y_pred = viterbi_decode(x, self.u, self.b_start, self.b_end, mask)
    nb_classes = self.input_spec[0].shape[2]
    y_pred_one_hot = K.one_hot(y_pred, nb_classes)
    return K.in_train_phase(x, y_pred_one_hot)

  def loss(self, y_true, y_pred):
    mask = self._fetch_mask()
    return chain_crf_loss(y_true, y_pred, self.u, self.b_start, self.b_end,
                          mask)

  def sparse_loss(self, y_true, y_pred):
    y_true = K.cast(y_true, 'int32')
    y_true = K.squeeze(y_true, 2)
    mask = self._fetch_mask()
    return sparse_chain_crf_loss(y_true, y_pred, self.u, self.b_start,
                                 self.b_end, mask)

  def get_config(self):
    config = {
        'init': initializers.serialize(self.init),
        'U_regularizer': regularizers.serialize(self.U_regularizer),
        'b_start_regularizer': regularizers.serialize(self.
                                                      b_start_regularizer),
        'b_end_regularizer': regularizers.serialize(self.b_end_regularizer),
        'U_constraint': constraints.serialize(self.U_constraint),
        'b_start_constraint': constraints.serialize(self.
                                                    b_start_constraint),
        'b_end_constraint': constraints.serialize(self.b_end_constraint)
    }
    base_config = super(ChainCRF, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

def create_custom_objects():
  instance_holder = {'instance': None}

  class ChainCRFClassWrapper(ChainCRF):
    def __init__(self, *args, **kwargs):
      instance_holder['instance'] = self
      super(ChainCRFClassWrapper, self).__init__(*args, **kwargs)

  def loss(*args):
    method = getattr(instance_holder['instance'], 'loss')
    return method(*args)

  def sparse_loss(*args):
    method = getattr(instance_holder['instance'], 'sparse_loss')
    return method(*args)

  return {'ChainCRF': ChainCRFClassWrapper,
          'ChainCRFClassWrapper': ChainCRFClassWrapper, 'loss': loss,
          'sparse_loss': sparse_loss}