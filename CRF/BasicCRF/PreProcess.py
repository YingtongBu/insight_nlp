#!/usr/bin/env python3
#coding: utf8
#author: Xinyi Wu (xinyi.wu5@pactera.com)
import pycrfsuite

def word2features(doc, i, model):
  word = doc[i][0]
  postag = doc[i][1]

  # Common features for all words
  features = [
    'bias',
    'word=' + word,
    'word[-4:]=' + word[-4:],
    'word[-3:]=' + word[-3:],
    'word[-2:]=' + word[-2:],
    'word[-1:]=' + word[-1:],
    'word.isdigit=%s' % word.isdigit(),
    'postag=' + postag,
    "word.gongsi={}".format(bool("公司" in word)),
  ]

  if model == "company":
    features.extend(["word.gongsi={}".format(bool("公司" in word))])
  elif model == "contract":
    features.extend(['word.hetong={}'.format(bool("合同" in word)),
                     'word.xieyi={}'.format(bool("协议" in word))])
  elif model == "project":
    features.extend(['word.xiangmu={}'.format(bool("项目" in word)),
                     'word.gongcheng={}'.format(bool("工程" in word))
                     ])

  # Features for words that are not at the beginning of a document
  if i > 0:
    word1 = doc[i-1][0]
    postag1 = doc[i-1][1]
    features.extend([
      '-1:word=' + word1,
      '-1:word.isdigit=%s' % word1.isdigit(),
      '-1:postag=' + postag1
    ])

    if model == "company":
      features.extend(["-1:word.shoudao={}".format(bool("收到" in word1))])
    elif model == "contract":
      features.extend(['-1:word.left={}'.format(bool("《" in word1)),
                      '-1:word.qianding={}'.format(bool("签订" in word1))])
    elif model == "project":
      features.extend(['-1:word.wei={}'.format(bool("为" in word1)),
                       '-1:word.zhongbiao={}'.format(bool("中标" in word1))
                       ])

    if i > 1:
      word2 = doc[i-2][0]
      postag2 = doc[i-2][1]
      features.extend([
        '-2:word=' + word2,
        '-2:word.isdigit=%s' % word2.isdigit(),
        '-2:postag=' + postag2
      ])

      if model == "company":
        features.extend(["-2:word.shoudao={}".format(bool("收到" in word2))])
      elif model == "contract":
        features.extend(['-2:word.left={}'.format(bool("《" in word2)),
                         '-2:word.qianding={}'.format(bool("签订" in word2))])
      elif model == "project":
        features.extend(['-2:word.wei={}'.format(bool("为" in word2))])

    else:
      features.append('SBOS')
  else:
    # Indicate that it is the 'beginning of a document'
    features.append('BOS')

  # Features for words that are not at the end of a document
  if i < len(doc)-1:
    word1 = doc[i+1][0]
    postag1 = doc[i+1][1]
    features.extend([
      '+1:word=' + word1,
      '+1:word.isdigit=%s' % word1.isdigit(),
      '+1:postag=' + postag1,
      '+1:word.fa={}'.format(bool("发" in word1))
    ])

    if model == "company":
      features.extend(["+1:word.fa={}".format(bool("发" in word1))])
    elif model == "contract":
      features.extend(['+1:word.right={}'.format(bool("》" in word1)),
                       '+1:word.hetong={}'.format(bool("合同" in word1)),
                       '+1:word.xieyi={}'.format(bool("协议" in word1))])
    elif model == "project":
      features.extend(['+1:word.zhongbiao={}'.format(bool("中标" in word1))
                       ])

    if i < len(doc)-2:
      word2 = doc[i+2][0]
      postag2 = doc[i+2][1]
      features.extend([
        '+2:word=' + word2,
        '+2:word.isdigit=%s' % word2.isdigit(),
        '+2:postag=' + postag2
      ])

      if model == "company":
        features.extend(["+2:word.fa={}".format(bool("发" in word2))])
      elif model == "contract":
        features.extend(['+2:word.right={}'.format(bool("》" in word2)),
                         '+2:word.hetong={}'.format(bool("合同" in word2)),
                         '+2:word.xieyi={}'.format(bool("协议" in word2))])
      elif model == "project":
        features.extend(['+2:word.zhongbiao={}'.format(bool("中标" in word2))])

    else:
      features.append('SEOE')
  else:
    # Indicate that it is the 'end of a document'
    features.append('EOS')

  return features

# A function for extracting features in documents
def extract_features(doc, model):
  return [word2features(doc, i, model) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
  return [label for (token, postag, label) in doc]

def train(X_train, y_train, model_name):
  # Training
  trainer = pycrfsuite.Trainer(verbose=False)

  # Submit training data to the trainer
  for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

  # Set the parameters of the model
  trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,

    # maximum number of iterations
    'max_iterations': 200,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
  })

  # Provide a file name as a parameter to the train function, such that
  # the model will be saved to the file when training is finished
  trainer.train(model_name)

  return None

def get_longest(X, Y):
  n = len(Y)
  current_len = 0  # To store the length of current substring
  max_len = 0  # To store the result
  prev_index = -2  # To store the previous index
  current_start = -1
  max_start = -1

  for i in range(n):
    if Y[i] == "Y":
      if prev_index == i - 1:
        current_len += 1
        prev_index = i
      else:
        prev_index = i
        current_len = 1
        current_start = i
      if current_len > max_len:
        max_len = current_len
        max_start = current_start

  if max_start == -1:
    return ''
  else:
    return ''.join(X[max_start:(max_start+max_len)])