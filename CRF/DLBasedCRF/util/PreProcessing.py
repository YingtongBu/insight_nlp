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
from .WordEmbeddings import word_normalize
from .CoNLL import read_co_nll
import sys

if (sys.version_info > (3, 0)):
  import pickle as pkl
else:  
  import cPickle as pkl
  from io import open

def prepare_dataset(embeddingsPath, datasets,
                    frequencyThresholdUnknownTokens=50,
                    reducePretrainedEmbeddings=False, valTransformations=None,
                    padOneTokenSentence=True):
  embeddingsName = os.path.splitext(embeddingsPath)[0]
  pklName = "_".join(sorted(datasets.keys()) + [embeddingsName])
  outputPath = 'pkl/' + pklName + '.pkl'

  if os.path.isfile(outputPath):
    logging.info("Using existent pickle file: %s" % outputPath)
    return outputPath

  casing2Idx = get_casing_vocab()
  embeddings, word2Idx = read_embeddings(embeddingsPath, datasets,
                                         frequencyThresholdUnknownTokens,
                                         reducePretrainedEmbeddings)
    
  mappings = {'tokens': word2Idx, 'casing': casing2Idx}
  pklObjects = {'embeddings': embeddings, 'mappings': mappings, 
                'datasets': datasets, 'data': {}}

  for datasetName, dataset in datasets.items():
    datasetColumns = dataset['columns']
    commentSymbol = dataset['commentSymbol']

    trainData = 'DataForModelTraining/%s/train.txt' % datasetName 
    devData = 'DataForModelTraining/%s/validation.txt' % datasetName 
    testData = 'DataForModelTraining/%s/test.txt' % datasetName 
    paths = [trainData, devData, testData]

    logging.info(":: Transform " + datasetName + " dataset ::")
    pklObjects['data'][datasetName] = create_pkl_files(paths, mappings,
                                                       datasetColumns,
                                                       commentSymbol,
                                                       valTransformations,
                                                       padOneTokenSentence)

  f = open(outputPath, 'wb')
  pkl.dump(pklObjects, f, -1)
  f.close()
    
  logging.info("DONE - Embeddings file saved: %s" % outputPath)
    
  return outputPath

def load_dataset_pickle(embeddingsPickle):
  f = open(embeddingsPickle, 'rb')
  pklObjects = pkl.load(f)
  f.close()

  return pklObjects['embeddings'], pklObjects['mappings'], pklObjects['data']

def read_embeddings(embeddingsPath, datasetFiles,
                    frequencyThresholdUnknownTokens,
                    reducePretrainedEmbeddings):
  if not os.path.isfile(embeddingsPath):
    if embeddingsPath in ['komninos_english_embeddings.gz', 
                          'levy_english_dependency_embeddings.gz', 
                          'reimers_german_embeddings.gz']:
      get_embeddings(embeddingsPath)
    else:
      print("The embeddings file %s was not found" % embeddingsPath)
      exit()

  logging.info("Generate new embeddings files for a dataset")

  neededVocab = {}
  if reducePretrainedEmbeddings:
    logging.info("Compute which tokens are required for the experiment")

    def create_dict(filename, tokenPos, vocab):
      for line in open(filename):
        if line.startswith('#'):
          continue
        splits = line.strip().split()
        if len(splits) > 1:
          word = splits[tokenPos]
          wordLower = word.lower()
          wordNormalized = word_normalize(wordLower)

          vocab[word] = True
          vocab[wordLower] = True
          vocab[wordNormalized] = True

    for dataset in datasetFiles:
      dataColumnsIdx = {y: x for x, y in dataset['cols'].items()}
      tokenIdx = dataColumnsIdx['tokens']
      datasetPath = 'DataForModelTraining/%s/' % dataset['name']

      for dataset in ['train.txt', 'validation.txt', 'test.txt']:
        create_dict(datasetPath + dataset, tokenIdx, neededVocab)

  logging.info("Read file: %s" % embeddingsPath)
  word2Idx = {}
  embeddings = []

  if embeddingsPath.endswith('.gz'):
    embeddingsIn = gzip.open(embeddingsPath, "rt")
  else:
    embeddingsIn = open(embeddingsPath, encoding="utf8")

  embeddingsDimension = None

  for line in embeddingsIn:
    split = line.rstrip().split(" ")
    word = split[0]

    if embeddingsDimension is None:
      embeddingsDimension = len(split) - 1

    if (len(split) - 1) != embeddingsDimension:  
      print("ERROR: A line in the embeddings file had more or less", 
            "dimensions than expected. Skip token.")
      continue

    if len(word2Idx) == 0:  
      word2Idx["PADDING_TOKEN"] = len(word2Idx)
      vector = np.zeros(embeddingsDimension)
      embeddings.append(vector)
      word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
      vector = np.random.uniform(-0.25, 0.25, embeddingsDimension)  
      embeddings.append(vector)

    vector = np.array([float(num) for num in split[1:]])

    if len(neededVocab) == 0 or word in neededVocab:
      if word not in word2Idx:
        embeddings.append(vector)
        word2Idx[word] = len(word2Idx)

  def create_fd(filename, tokenIndex, fd, word2Idx):
    for line in open(filename):
      if line.startswith('#'):
        continue

      splits = line.strip().split()

      if len(splits) > 1:
        word = splits[tokenIndex]
        wordLower = word.lower()
        wordNormalized = word_normalize(wordLower)

        if (word not in word2Idx and wordLower not in word2Idx and 
           wordNormalized not in word2Idx):
          fd[wordNormalized] += 1

  if (frequencyThresholdUnknownTokens is not None and 
     frequencyThresholdUnknownTokens >= 0):
    fd = nltk.FreqDist()
    for datasetName, datasetFile in datasetFiles.items():
      dataColumnsIdx = {y: x for x, y in datasetFile['columns'].items()}
      tokenIdx = dataColumnsIdx['tokens']
      datasetPath = 'DataForModelTraining/%s/' % datasetName
      create_fd(datasetPath + 'train.txt', tokenIdx, fd, word2Idx)

    addedWords = 0
    for word, freq in fd.most_common(10000):
      if freq < frequencyThresholdUnknownTokens:
        break

      addedWords += 1
      word2Idx[word] = len(word2Idx)
      vector = np.random.uniform(-0.25, 0.25, len(split) - 1)  
      embeddings.append(vector)

      assert (len(word2Idx) == len(embeddings))

    logging.info("Added words: %d" % addedWords)
  embeddings = np.array(embeddings)

  return embeddings, word2Idx

def add_char_information(sentences):
  for sentenceIdx in range(len(sentences)):
    sentences[sentenceIdx]['characters'] = []
    for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
      token = sentences[sentenceIdx]['tokens'][tokenIdx]
      chars = [c for c in token]
      sentences[sentenceIdx]['characters'].append(chars)

def add_casing_information(sentences):
  for sentenceIdx in range(len(sentences)):
    sentences[sentenceIdx]['casing'] = []
    for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
      token = sentences[sentenceIdx]['tokens'][tokenIdx]
      sentences[sentenceIdx]['casing'].append(get_casing(token))
  return sentences[sentenceIdx]['casing']
       
def get_casing(word):
  casing = 'other'
    
  numDigits = 0
  for char in word:
    if char.isdigit():
      numDigits += 1
            
  digitFraction = numDigits / float(len(word))
    
  if word.isdigit():  
    casing = 'numeric'
  elif digitFraction > 0.5:
    casing = 'mainly_numeric'
  elif word.islower():  
    casing = 'allLower'
  elif word.isupper():  
    casing = 'allUpper'
  elif word[0].isupper():  
    casing = 'initialUpper'
  elif numDigits > 0:
    casing = 'contains_digit'
    
  return casing

def get_casing_vocab():
  entries = ['PADDING', 'other', 'numeric', 'mainly_numeric', 'allLower', 
             'allUpper', 'initialUpper', 'contains_digit']
  return {entries[idx]: idx for idx in range(len(entries))}

def create_matrices(sentences, mappings, padOneTokenSentence):
  data = []
  numTokens = 0
  numUnknownTokens = 0    
  missingTokens = FreqDist()
  paddedSentences = 0

  for sentence in sentences:
    row = {name: [] for name in list(mappings.keys()) + ['raw_tokens']}
        
    for mapping, str2Idx in mappings.items():    
      if mapping not in sentence:
        continue
                    
      for entry in sentence[mapping]:                
        if mapping.lower() == 'tokens':
          numTokens += 1
          idx = str2Idx['UNKNOWN_TOKEN']
                    
          if entry in str2Idx:
            idx = str2Idx[entry]
          elif entry.lower() in str2Idx:
            idx = str2Idx[entry.lower()]
          elif word_normalize(entry) in str2Idx:
            idx = str2Idx[word_normalize(entry)]
          else:
            numUnknownTokens += 1    
            missingTokens[word_normalize(entry)] += 1
                        
          row['raw_tokens'].append(entry)
        elif mapping.lower() == 'characters':  
          idx = []
          for c in entry:
            if c in str2Idx:
              idx.append(str2Idx[c])
            else:
              idx.append(str2Idx['UNKNOWN'])                           
                                      
        else:
          idx = str2Idx[entry]
                                    
        row[mapping].append(idx)
                
    if len(row['tokens']) == 1 and padOneTokenSentence:
      paddedSentences += 1
      for mapping, str2Idx in mappings.items():
        if mapping.lower() == 'tokens':
          row['tokens'].append(mappings['tokens']['PADDING_TOKEN'])
          row['raw_tokens'].append('PADDING_TOKEN')
        elif mapping.lower() == 'characters':
          row['characters'].append([0])
        else:
          row[mapping].append(0)
            
    data.append(row)
    
  if numTokens > 0:           
    logging.info("Unknown-Tokens: %.2f%%" % 
                 (numUnknownTokens / float(numTokens) * 100))
        
  return data

def create_pkl_files(datasetFiles, mappings, cols, commentSymbol,
                     valTransformation, padOneTokenSentence):
  trainSentences = read_co_nll(datasetFiles[0], cols, commentSymbol,
                               valTransformation)
  devSentences = read_co_nll(datasetFiles[1], cols, commentSymbol,
                             valTransformation)
  testSentences = read_co_nll(datasetFiles[2], cols, commentSymbol,
                              valTransformation)
   
  extend_mappings(mappings, trainSentences + devSentences + testSentences)
  charset = {"PADDING": 0, "UNKNOWN": 1}
  for c in (" 0123456789abcdefghijklmnopqrstuvwxyz", 
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|"):
    charset[c] = len(charset)
  mappings['characters'] = charset
    
  add_char_information(trainSentences)
  add_casing_information(trainSentences)
    
  add_char_information(devSentences)
  add_casing_information(devSentences)
    
  add_char_information(testSentences)
  add_casing_information(testSentences)

  logging.info(":: Create Train Matrix ::")
  trainMatrix = create_matrices(trainSentences, mappings, padOneTokenSentence)

  logging.info(":: Create Dev Matrix ::")
  devMatrix = create_matrices(devSentences, mappings, padOneTokenSentence)

  logging.info(":: Create Test Matrix ::")
  testMatrix = create_matrices(testSentences, mappings, padOneTokenSentence)
    
  data = {
      'trainMatrix': trainMatrix,
      'devMatrix': devMatrix,
      'testMatrix': testMatrix
  }        
    
  return data

def extend_mappings(mappings, sentences):
  sentenceKeys = list(sentences[0].keys())
  sentenceKeys.remove('tokens') 

  for sentence in sentences:
    for name in sentenceKeys:
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
    metaFunc = meta.getheaders if hasattr(
        meta, 'getheaders') else meta.get_all
    metaLength = metaFunc('Content-Length')
    try:
      return int(metaLength[0])
    except:
      return 0

  def kb_to_mb(kb):
    return kb / 1024.0 / 1024.0

  def callback(blocks, blockSize, totalSize):
    current = blocks * blockSize
    percent = 100.0 * current / totalSize
    line = '[{0}{1}]'.format(
        '=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
    status = '\r{0:3.0f}%{1} {2:3.1f}/{3:3.1f} MB'
    sys.stdout.write(
        status.format(
            percent, line, kb_to_mb(current), kb_to_mb(totalSize)))

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