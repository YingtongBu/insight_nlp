#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)
import Insight_NLP.CRF.DLBasedCRF.DataProcessing as DataProcessing
import optparse
import Insight_NLP.CRF.DLBasedCRF.TrainModel as TrainModel
import Insight_NLP.CRF.DLBasedCRF.RunModel as RunModel

def main(word_vector_file, train_jia_fang_data_file, test_jia_fang_data_file,
         train_set_model_file, test_set_model_file, validation_set_model_file,
         predict_prob_file, id_record_file, input_file, output_file,
         ground_truth_file):
  DataProcessing.preprocess(word_vector_file, train_jia_fang_data_file,
                            test_jia_fang_data_file, train_set_model_file,
                            test_set_model_file, validation_set_model_file,
                            predict_prob_file, id_record_file, input_file)
  TrainModel.train_model()
  RunModel.run_model(input_file, output_file)
  DataProcessing.postprocess(id_record_file, output_file, ground_truth_file)
  
if __name__ == '__main__':
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option('-w', '--word_vector_file',
                    default='./chinese_word_vector')
  parser.add_option('-r', '--train_jia_fang_data_file',
                    default='./Data/JiaFang/trainTemp1.data')
  parser.add_option('-e', '--test_jia_fang_data_file',
                    default='./Data/JiaFang/testTemp1.data')
  parser.add_option('-t', '--train_set_model_file',
                    default='./DataForModelTraining/Chinese/train.data')
  parser.add_option('-x', '--test_set_model_file',
                    default='./DataForModelTraining/Chinese/test.data')
  parser.add_option('-v', '--validation_set_model_file',
                    default='./DataForModelTraining/Chinese/validation.data')
  parser.add_option('-p', '--predict_prob_file',
                    default='./Data/JiaFang/predictProb1.data')
  parser.add_option('-d', '--id_record_file',
                    default='./idRecord.txt')
  parser.add_option('-i', '--input_file',
                    default='./input.txt')
  parser.add_option('-o', '--output_file',
                    default='./output.txt')
  parser.add_option('-g', '--ground_truth_file',
                    default='./Data/label.test_lower.data')
  (options, args) = parser.parse_args()

  main(options.word_vector_file, options.train_jia_fang_data_file,
       options.test_jia_fang_data_file, options.train_set_model_file,
       options.test_set_model_file, options.validation_set_model_file,
       options.predict_prob_file, options.id_record_file,
       options.input_file, options.output_file, options.ground_truth_file)