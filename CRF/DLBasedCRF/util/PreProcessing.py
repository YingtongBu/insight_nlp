#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)
from __future__ import (division, absolute_import, print_function, 
                        unicode_literals)
import os
import numpy as np
import gzip
import os.path
import nltk
import logging
from nltk import FreqDist
from CRF.DLBasedCRF.util.WordEmbeddings import word_normalize
from CRF.DLBasedCRF.util.CoNLL import read_co_nll
import sys

if (sys.version_info > (3, 0)):
  import pickle as pkl
else:  
  import cPickle as pkl
  from io import open

def prepare_dataset(embeddings_path, datasets,
                    frequency_threshold_unknown_tokens=50,
                    reduce_pretrained_embeddings=False,
                    val_transformations=None,
                    pad_one_token_sentence=True):
  embeddings_name = os.path.splitext(embeddings_path)[0]
  pkl_name = "_".join(sorted(datasets.keys()) + [embeddings_name])
  output_path = 'pkl/' + pkl_name + '.pkl'

  if os.path.isfile(output_path):
    logging.info("Using existent pickle file: %s" % output_path)
    return output_path

  casing_to_idx = get_casing_vocab()
  embeddings, word_to_idx = read_embeddings(embeddings_path, datasets,
                                         frequency_threshold_unknown_tokens,
                                         reduce_pretrained_embeddings)
    
  mappings = {'tokens': word_to_idx, 'casing': casing_to_idx}
  pkl_objects = {'embeddings': embeddings, 'mappings': mappings,
                'datasets': datasets, 'data': {}}

  for dataset_name, dataset in datasets.items():
    dataset_columns = dataset['columns']
    comment_symbol = dataset['commentSymbol']

    train_data = 'DataForModelTraining/%s/train.data' % dataset_name
    dev_data = 'DataForModelTraining/%s/validation.data' % dataset_name
    test_data = 'DataForModelTraining/%s/test.data' % dataset_name
    paths = [train_data, dev_data, test_data]

    logging.info(":: Transform " + dataset_name + " dataset ::")
    pkl_objects['data'][dataset_name] = create_pkl_files(paths, mappings,
                                                       dataset_columns,
                                                       comment_symbol,
                                                       val_transformations,
                                                       pad_one_token_sentence)
  os.mkdir('./pkl')
  f = open(output_path, 'wb')
  pkl.dump(pkl_objects, f, -1)
  f.close()
    
  logging.info("DONE - Embeddings file saved: %s" % output_path)
    
  return output_path

def load_dataset_pickle(embeddings_pickle):
  f = open(embeddings_pickle, 'rb')
  pkl_objects = pkl.load(f)
  f.close()

  return pkl_objects['embeddings'], pkl_objects['mappings'], pkl_objects['data']

def read_embeddings(embeddings_path, dataset_files,
                    frequency_threshold_unknown_tokens,
                    reduce_pretrained_embeddings):
  if not os.path.isfile(embeddings_path):
    if embeddings_path in ['komninos_english_embeddings.gz',
                          'levy_english_dependency_embeddings.gz', 
                          'reimers_german_embeddings.gz']:
      get_embeddings(embeddings_path)
    else:
      print("The embeddings file %s was not found" % embeddings_path)
      exit()

  logging.info("Generate new embeddings files for a dataset")

  needed_vocab = {}
  if reduce_pretrained_embeddings:
    logging.info("Compute which tokens are required for the experiment")

    def create_dict(filename, token_pos, vocab):
      for line in open(filename):
        if line.startswith('#'):
          continue
        splits = line.strip().split()
        if len(splits) > 1:
          word = splits[token_pos]
          word_lower = word.lower()
          word_normalized = word_normalize(word_lower)

          vocab[word] = True
          vocab[word_lower] = True
          vocab[word_normalized] = True

    for dataset in dataset_files:
      data_columns_idx = {y: x for x, y in dataset['cols'].items()}
      token_idx = data_columns_idx['tokens']
      dataset_path = 'DataForModelTraining/%s/' % dataset['name']

      for dataset in ['train.data', 'validation.data', 'test.data']:
        create_dict(dataset_path + dataset, token_idx, needed_vocab)

  logging.info("Read file: %s" % embeddings_path)
  word_to_idx = {}
  embeddings = []

  if embeddings_path.endswith('.gz'):
    embeddings_in = gzip.open(embeddings_path, "rt")
  else:
    embeddings_in = open(embeddings_path, encoding="utf8")

  embeddings_dimension = None

  for line in embeddings_in:
    split = line.rstrip().split(" ")
    word = split[0]

    if embeddings_dimension is None:
      embeddings_dimension = len(split) - 1

    if (len(split) - 1) != embeddings_dimension:
      print("ERROR: A line in the embeddings file had more or less", 
            "dimensions than expected. Skip token.")
      continue

    if len(word_to_idx) == 0:
      word_to_idx["PADDING_TOKEN"] = len(word_to_idx)
      vector = np.zeros(embeddings_dimension)
      embeddings.append(vector)
      word_to_idx["UNKNOWN_TOKEN"] = len(word_to_idx)
      vector = np.random.uniform(-0.25, 0.25, embeddings_dimension)
      embeddings.append(vector)

    vector = np.array([float(num) for num in split[1:]])

    if len(needed_vocab) == 0 or word in needed_vocab:
      if word not in word_to_idx:
        embeddings.append(vector)
        word_to_idx[word] = len(word_to_idx)

  def create_fd(filename, token_index, fd, word_to_idx):
    for line in open(filename):
      if line.startswith('#'):
        continue

      splits = line.strip().split()

      if len(splits) > 1:
        word = splits[token_index]
        word_lower = word.lower()
        word_normalized = word_normalize(word_lower)

        if (word not in word_to_idx and word_lower not in word_to_idx and
           word_normalized not in word_to_idx):
          fd[word_normalized] += 1

  if (frequency_threshold_unknown_tokens is not None and
     frequency_threshold_unknown_tokens >= 0):
    fd = nltk.FreqDist()
    for dataset_name, dataset_file in dataset_files.items():
      data_columns_idx = {y: x for x, y in dataset_file['columns'].items()}
      token_idx = data_columns_idx['tokens']
      dataset_path = 'DataForModelTraining/%s/' % dataset_name
      create_fd(dataset_path + 'train.data', token_idx, fd, word_to_idx)

    added_words = 0
    for word, freq in fd.most_common(10000):
      if freq < frequency_threshold_unknown_tokens:
        break

      added_words += 1
      word_to_idx[word] = len(word_to_idx)
      vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
      embeddings.append(vector)

      assert (len(word_to_idx) == len(embeddings))

    logging.info("Added words: %d" % added_words)
  embeddings = np.array(embeddings)

  return embeddings, word_to_idx

def add_char_information(sentences):
  for sentence_idx in range(len(sentences)):
    sentences[sentence_idx]['characters'] = []
    for token_idx in range(len(sentences[sentence_idx]['tokens'])):
      token = sentences[sentence_idx]['tokens'][token_idx]
      chars = [c for c in token]
      sentences[sentence_idx]['characters'].append(chars)

def add_casing_information(sentences):
  for sentence_idx in range(len(sentences)):
    sentences[sentence_idx]['casing'] = []
    for token_idx in range(len(sentences[sentence_idx]['tokens'])):
      token = sentences[sentence_idx]['tokens'][token_idx]
      sentences[sentence_idx]['casing'].append(get_casing(token))
  return sentences[sentence_idx]['casing']
       
def get_casing(word):
  casing = 'other'
    
  num_digits = 0
  for char in word:
    if char.isdigit():
      num_digits += 1
            
  digit_fraction = num_digits / float(len(word))
    
  if word.isdigit():  
    casing = 'numeric'
  elif digit_fraction > 0.5:
    casing = 'mainly_numeric'
  elif word.islower():  
    casing = 'allLower'
  elif word.isupper():  
    casing = 'allUpper'
  elif word[0].isupper():  
    casing = 'initialUpper'
  elif num_digits > 0:
    casing = 'contains_digit'
    
  return casing

def get_casing_vocab():
  entries = ['PADDING', 'other', 'numeric', 'mainly_numeric', 'allLower', 
             'allUpper', 'initialUpper', 'contains_digit']
  return {entries[idx]: idx for idx in range(len(entries))}

def create_matrices(sentences, mappings, pad_one_token_sentence):
  data = []
  num_tokens = 0
  num_unknown_tokens = 0
  missing_tokens = FreqDist()
  padded_sentences = 0

  for sentence in sentences:
    row = {name: [] for name in list(mappings.keys()) + ['raw_tokens']}
        
    for mapping, str_to_idx in mappings.items():
      if mapping not in sentence:
        continue
                    
      for entry in sentence[mapping]:                
        if mapping.lower() == 'tokens':
          num_tokens += 1
          idx = str_to_idx['UNKNOWN_TOKEN']
                    
          if entry in str_to_idx:
            idx = str_to_idx[entry]
          elif entry.lower() in str_to_idx:
            idx = str_to_idx[entry.lower()]
          elif word_normalize(entry) in str_to_idx:
            idx = str_to_idx[word_normalize(entry)]
          else:
            num_unknown_tokens += 1
            missing_tokens[word_normalize(entry)] += 1
                        
          row['raw_tokens'].append(entry)
        elif mapping.lower() == 'characters':  
          idx = []
          for c in entry:
            if c in str_to_idx:
              idx.append(str_to_idx[c])
            else:
              idx.append(str_to_idx['UNKNOWN'])
                                      
        else:
          idx = str_to_idx[entry]
                                    
        row[mapping].append(idx)
                
    if len(row['tokens']) == 1 and pad_one_token_sentence:
      padded_sentences += 1
      for mapping, str_to_idx in mappings.items():
        if mapping.lower() == 'tokens':
          row['tokens'].append(mappings['tokens']['PADDING_TOKEN'])
          row['raw_tokens'].append('PADDING_TOKEN')
        elif mapping.lower() == 'characters':
          row['characters'].append([0])
        else:
          row[mapping].append(0)
            
    data.append(row)
    
  if num_tokens > 0:
    logging.info("Unknown-Tokens: %.2f%%" % 
                 (num_unknown_tokens / float(num_tokens) * 100))
        
  return data

def create_pkl_files(dataset_files, mappings, cols, comment_symbol,
                     val_transformation, pad_one_token_sentence):
  train_sentences = read_co_nll(dataset_files[0], cols, comment_symbol,
                               val_transformation)
  dev_sentences = read_co_nll(dataset_files[1], cols, comment_symbol,
                             val_transformation)
  test_sentences = read_co_nll(dataset_files[2], cols, comment_symbol,
                              val_transformation)
   
  extend_mappings(mappings, train_sentences + dev_sentences + test_sentences)
  charset = {"PADDING": 0, "UNKNOWN": 1}
  for c in (" 0123456789abcdefghijklmnopqrstuvwxyz", 
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|"):
    charset[c] = len(charset)
  mappings['characters'] = charset
    
  add_char_information(train_sentences)
  add_casing_information(train_sentences)
    
  add_char_information(dev_sentences)
  add_casing_information(dev_sentences)
    
  add_char_information(test_sentences)
  add_casing_information(test_sentences)

  logging.info(":: Create Train Matrix ::")
  train_matrix = create_matrices(train_sentences, mappings,
                                 pad_one_token_sentence)

  logging.info(":: Create Dev Matrix ::")
  dev_matrix = create_matrices(dev_sentences, mappings,
                               pad_one_token_sentence)

  logging.info(":: Create Test Matrix ::")
  test_matrix = create_matrices(test_sentences, mappings,
                                pad_one_token_sentence)
    
  data = {
      'trainMatrix': train_matrix,
      'devMatrix': dev_matrix,
      'testMatrix': test_matrix
  }        
    
  return data

def extend_mappings(mappings, sentences):
  sentence_keys = list(sentences[0].keys())
  sentence_keys.remove('tokens')

  for sentence in sentences:
    for name in sentence_keys:
      if name not in mappings:
        mappings[name] = {'O': 0} 

      for item in sentence[name]:              
        if item not in mappings[name]:
          mappings[name][item] = len(mappings[name])

def get_embeddings(name):
  if not os.path.isfile(name):
    download("https://public.ukp.informatik.tu-darmstadt.de/reimers/",
             "embeddings/" + name)

def get_levy_dependency_embeddings():
  if not os.path.isfile("levy_deps.words.bz2"):
    print("Start downloading word embeddings from Levy et al. ...")
    os.system("wget -O levy_deps.words.bz2 ",
              "http://u.cs.biu.ac.il/~yogo/DataForModelTraining/syntemb/",
              "deps.words.bz2")
    
  print("Start unzip word embeddings ...")
  os.system("bzip2 -d levy_deps.words.bz2")

def get_reimers_embeddings():
  if not os.path.isfile("2014_tudarmstadt_german_50mincount.vocab.gz"):
    print("Start downloading word embeddings from Reimers et al. ...")
    os.system("wget https://public.ukp.informatik.tu-darmstadt.de/reimers/",
              "2014_german_embeddings/2014_tudarmstadt_german_50mincount.",
              "vocab.gz")
    
if sys.version_info >= (3,):
  import urllib.request as urllib2
  import urllib.parse as urlparse
  from urllib.request import urlretrieve
else:
  import urllib2
  import urlparse
  from urllib import urlretrieve

def download(url, destination=os.curdir, silent=False):
  filename = (os.path.basename(urlparse.urlparse(url).path) or 
              'downloaded.file')

  def get_size():
    meta = urllib2.urlopen(url).info()
    meta_func = meta.getheaders if hasattr(
        meta, 'getheaders') else meta.get_all
    meta_length = meta_func('Content-Length')
    try:
      return int(meta_length[0])
    except:
      return 0

  def kb_to_mb(kb):
    return kb / 1024.0 / 1024.0

  def callback(blocks, block_size, total_size):
    current = blocks * block_size
    percent = 100.0 * current / total_size
    line = '[{0}{1}]'.format(
        '=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
    status = '\r{0:3.0f}%{1} {2:3.1f}/{3:3.1f} MB'
    sys.stdout.write(
        status.format(
            percent, line, kb_to_mb(current), kb_to_mb(total_size)))

  path = os.path.join(destination, filename)

  logging.info(
      'Downloading: {0} ({1:3.1f} MB)'.format(url, kb_to_mb(get_size())))
  try:
    (path, headers) = urlretrieve(url, path, None if silent else callback)
  except:
    os.remove(path)
    raise Exception("Can't download {0}".format(path))
  else:
    print()
    logging.info('Downloaded to: {0}'.format(path))

  return path