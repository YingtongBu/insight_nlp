#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

import optparse
import os
from Insight_NLP.Classifier.CNN.Train import preprocess
from Insight_NLP.Classifier.CNN.Train import train

def main(argv=None):
  x_train, y_train, vocab_processor, x_dev, y_dev, x_ori_dev = preprocess()
  train(x_train, y_train, vocab_processor, x_dev, y_dev, x_ori_dev)

if __name__ == '__main__':
  usage = 'usage = %prog [options]'
  parser = optparse.OptionParser(usage=usage)
  parser.add_option("--dev_sample_percentage", type=float, default=.3,
                    help='Percentage of the training data used for validation')
  parser.add_option("--train_data", type=str, default='TotalEvents.data',
                    help='Data source for training.')
  parser.add_option("--num_classes", type=int, default=44,
                    help='number of classes going to be classified')
  parser.add_option("--embedding_dim", type=int, default=128,
                    help='Dimensionality of character embedding(default: 128)')
  parser.add_option("--kernel_sizes", type=str, default='1,1,1,2,3',
                    help='Comma-separated kernel sizes (default: 3,4,5)')
  parser.add_option("--num_kernels", type=int, default=128,
                    help='Number of filters per filter size (default: 128)')
  parser.add_option("--dropout_keep_prob", type=float, default=.5,
                    help='Dropout keep probability (default: 0.5)')
  parser.add_option("--l2_reg_lambda", type=float, default=0.0,
                      help='L2 regularization lambda (default: 0.0)')
  parser.add_option("--num_words", type=int, default=64,
                    help='Number of words kept in each sentence (default: 64)')
  parser.add_option("--batch_size", type=int, default=1024,
                    help='Batch Size (default: 64)')
  parser.add_option("--num_epochs", type=int, default=2,
                    help='Number of training epochs (default: 2)')
  parser.add_option("--evaluate_every", type=int, default=100,
                    help='Evaluate model on dev every # steps (default: 100)')
  parser.add_option("--language_type", type=str, default='ENG',
                    help='Language type of input data, [CHI, ENG]')
  parser.add_option("--GPU", type=str, default='3',
                    help='GPU device you use')
  options, args = parser.parse_args()

  os.environ["CUDA_VISIBLE_DEVICES"] = options.GPU
  main()