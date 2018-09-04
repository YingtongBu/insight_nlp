#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)
import optparse
from Insight_NLP.CRF.BasicCRF.FeatureExtraction import FeatureExtraction
from Insight_NLP.CRF.BasicCRF.CRFPredictor import CRFPredictor

class SpecificFeatureExtraction(FeatureExtraction):
  def _word_to_features(self, data, i):
    word = data[i][0]
    postag = data[i][1]

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

    features.extend(["word.gongsi={}".format(bool("公司" in word))])

    # Features for words that are not at the beginning of a document
    if i > 0:
      word1 = data[i - 1][0]
      postag1 = data[i - 1][1]
      features.extend([
        '-1:word=' + word1,
        '-1:word.isdigit=%s' % word1.isdigit(),
        '-1:postag=' + postag1
      ])

      features.extend(["-1:word.shoudao={}".format(bool("收到" in word1))])

      if i > 1:
        word2 = data[i - 2][0]
        postag2 = data[i - 2][1]
        features.extend([
          '-2:word=' + word2,
          '-2:word.isdigit=%s' % word2.isdigit(),
          '-2:postag=' + postag2
        ])

        features.extend(["-2:word.shoudao={}".format(bool("收到" in word2))])

      else:
        features.append('SBOS')
    else:
      # Indicate that it is the 'beginning of a document'
      features.append('BOS')

    # Features for words that are not at the end of a document
    if i < len(data) - 1:
      word1 = data[i + 1][0]
      postag1 = data[i + 1][1]
      features.extend([
        '+1:word=' + word1,
        '+1:word.isdigit=%s' % word1.isdigit(),
        '+1:postag=' + postag1,
        '+1:word.fa={}'.format(bool("发" in word1))
      ])

      features.extend(["+1:word.fa={}".format(bool("发" in word1))])

      if i < len(data) - 2:
        word2 = data[i + 2][0]
        postag2 = data[i + 2][1]
        features.extend([
          '+2:word=' + word2,
          '+2:word.isdigit=%s' % word2.isdigit(),
          '+2:postag=' + postag2
        ])

        features.extend(["+2:word.fa={}".format(bool("发" in word2))])

      else:
        features.append('SEOE')
    else:
      # Indicate that it is the 'end of a document'
      features.append('EOS')

    return features

if __name__ == '__main__':

  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option('-m', '--model_name',
                    default='company')
  parser.add_option('-d', '--doc_path',
                    default='./DataForTaskTest/JiaFang/testTempJiafang.data')
  parser.add_option('-o', '--output_file',
                    default='./output.txt')
  (options, args) = parser.parse_args()

  feature_extraction = SpecificFeatureExtraction(options.doc_path)
  crf_predictor = CRFPredictor(options.model_name, options.doc_path,
                           feature_extraction, options.output_file)
  crf_predictor.crf_predictor()