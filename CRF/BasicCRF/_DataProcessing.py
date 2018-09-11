#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

from Insight_NLP.Chinese import segment_sentence
from Insight_NLP.Common import read_pydict_file

class _DataProcessing(object):

  def __init__(self, data_file):
    self.data_file = data_file

  def _process_train_data(self):
    data = read_pydict_file(self.data_file)
    sample = list()

    for line in data:
      tags = line['tags']
      text = line['text']
      part_pos_text = []
      tags.sort()

      for pos in range(len(tags)):
        if text[tags[pos][0]:tags[pos][1]] == tags[pos][3]:
          if pos == 0 and tags[pos][0] != 0:

            pre_result_words, pre_result_tags = \
              segment_sentence(text[0:tags[pos][0]], True)
          else:
            pre_result_words, pre_result_tags = \
              segment_sentence(text[tags[pos - 1][1]:tags[pos][0]], True)
          result_words, result_tags = \
            segment_sentence(text[tags[pos][0]:tags[pos][1]], True)
          for i in range(len(pre_result_words)):
            part_pos_text.append([pre_result_words[i], pre_result_tags[i], 'O'])
          part_pos_text.append([result_words[0], result_tags[0], 'B-' +
                                tags[pos][2]])
          if len(result_tags) == 1:
            continue
          else:
            i = 1
            while i < len(result_tags):
              part_pos_text.append([result_words[i], result_tags[i],
                                    'I-' + tags[pos][2]])
              i += 1
        else:
          print('ERROR in Tags information!! Please check train.pydict file!')

      sample.append(part_pos_text)
    return sample

  def _process_test_data(self):
    words, tags = segment_sentence(self.data_file, True)
    sample = list()
    for pos in range(len(words)):
      sample.append([words[pos], tags[pos]])
    return [sample]
  
  #code review
  def _process_line(self, ...):

  #code review:
  def _process_test_data_batch(self, file_name):
    doc_file = open(self.data_file)
    sample = list()
    for line in doc_file:
      words, tags = segment_sentence(line, True)
      line_sample = list()
      for pos in range(len(words)):
        line_sample.append([words[pos], tags[pos]])
      sample.append(line_sample)
    doc_file.close()
    return sample

  def _get_labels(self, sample):
    return [label for (token, postag, label) in sample]

  def _get_longest_label(self, X, Y):
    n = len(Y)
    current_len = 0
    max_len = 0
    prev_index = -2
    current_start = -1
    max_start = -1

    for i in range(n):
      if Y[i].startwith('B'):
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
      return ''.join(X[max_start:(max_start + max_len)])