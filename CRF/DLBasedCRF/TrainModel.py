#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

from __future__ import print_function
import os
import logging
import sys
from CRF.DLBasedCRF.NeuralNets.BiLSTM import BiLSTM
from CRF.DLBasedCRF.util.PreProcessing \
  import prepare_dataset, load_dataset_pickle
from keras import backend as K

def train_model():
  abspath = os.path.abspath(__file__)
  dname = os.path.dirname(abspath)
  os.chdir(dname)

  logging_level = logging.INFO
  logger = logging.getLogger()
  logger.setLevel(logging_level)

  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging_level)
  formatter = logging.Formatter('%(message)s')
  ch.setFormatter(formatter)
  logger.addHandler(ch)

  datasets = {
      'Chinese': {
          'columns': {1: 'tokens', 2: 'NER_BIO'},
          'label': 'NER_BIO',
          'evaluate': True,
          'commentSymbol': None
      }
  }

  embeddings_path = 'chinese_word_vector'
  pickle_file = prepare_dataset(embeddings_path, datasets)
  embeddings, mappings, data = load_dataset_pickle(pickle_file)
  params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 
            'dropout': (0.25, 0.25), 'charEmbeddings': None, 
            'maxCharLength': 50}

  model = BiLSTM(params)
  model.set_mappings(mappings, embeddings)
  model.set_dataset(datasets, data)
  model.store_results('results/Chinese_NER_results.csv')
  model.model_save_path = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
  model.fit(epochs=25)