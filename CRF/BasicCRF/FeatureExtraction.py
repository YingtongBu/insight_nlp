#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

class FeatureExtraction(object):

  def _extract_features(self, data):
    return [self.word_to_features(data, i)
            for i in range(len(data))]

  # @override
  def word_to_features(self, sample, pos):
    '''
    :param sample: a data sample
    :param pos: the position where we need to do word2feature process
    :return: features extracted from input data
    '''
    word = sample[pos][0]
    postag = sample[pos][1]

    features = [
      'bias',
      'word=' + word,
      'word[-4:]=' + word[-4:],
      'word[-3:]=' + word[-3:],
      'word[-2:]=' + word[-2:],
      'word[-1:]=' + word[-1:],
      'word.isdigit=%s' % word.isdigit(),
      'postag=' + postag
    ]

    if pos > 0:
      word1 = sample[pos - 1][0]
      postag1 = sample[pos - 1][1]
      features.extend([
        '-1:word=' + word1,
        '-1:word.isdigit=%s' % word1.isdigit(),
        '-1:postag=' + postag1
      ])

      if pos > 1:
        word2 = sample[pos - 2][0]
        postag2 = sample[pos - 2][1]
        features.extend([
          '-2:word=' + word2,
          '-2:word.isdigit=%s' % word2.isdigit(),
          '-2:postag=' + postag2
        ])
      else:
        features.append('SBOS')
    else:
      features.append('BOS')

    if pos < len(sample) - 1:
      word1 = sample[pos + 1][0]
      postag1 = sample[pos + 1][1]
      features.extend([
        '+1:word=' + word1,
        '+1:word.isdigit=%s' % word1.isdigit(),
        '+1:postag=' + postag1
      ])

      if pos < len(sample) - 2:
        word2 = sample[pos + 2][0]
        postag2 = sample[pos + 2][1]
        features.extend([
          '+2:word=' + word2,
          '+2:word.isdigit=%s' % word2.isdigit(),
          '+2:postag=' + postag2
        ])

      else:
        features.append('SEOE')
    else:
      features.append('EOS')

    return features