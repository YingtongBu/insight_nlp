#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

import DataProcessing
import optparse
import TrainModel
import RunModel

def main(wordVectorFile, trainJiaFangDataFile, testJiaFangDataFile, 
         trainSetModelFile, testSetModelFile, validationSetModelFile,
         predictProbFile, idRecordFile, inputFile, outputFile, groundTruthFile):
  DataProcessing.preprocess(wordVectorFile, trainJiaFangDataFile, 
                            testJiaFangDataFile, trainSetModelFile,
                            testSetModelFile, validationSetModelFile,
                            predictProbFile, idRecordFile, inputFile)
  TrainModel.trainModel()
  RunModel.runModel(inputFile, outputFile)
  DataProcessing.postprocess(idRecordFile, outputFile, groundTruthFile)
  
if __name__ == '__main__':
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option('-w', '--wordVectorFile', 
                    default='./chineseWordVector')
  parser.add_option('-r', '--trainJiaFangDataFile', 
                    default='./DataForTaskTest/JiaFang/trainTemp1.data')
  parser.add_option('-e', '--testJiaFangDataFile', 
                    default='./DataForTaskTest/JiaFang/testTemp1.temp')
  parser.add_option('-t', '--trainSetModelFile', 
                    default='./DataForModelTraining/Chinese/train.txt')
  parser.add_option('-x', '--testSetModelFile', 
                    default='./DataForModelTraining/Chinese/test.txt')
  parser.add_option('-v', '--validationSetModelFile', 
                    default='./DataForModelTraining/Chinese/validation.txt')
  parser.add_option('-p', '--predictProbFile', 
                    default='./DataForTaskTest/JiaFang/predictProb1.temp') 
  parser.add_option('-d', '--idRecordFile', 
                    default='./idRecord.txt')
  parser.add_option('-i', '--inputFile', 
                    default='./input.txt')
  parser.add_option('-o', '--outputFile', 
                    default='./outputFile.txt')
  parser.add_option('-g', '--groundTruthFile', 
                    default='./DataForTaskTest/label.test_lower.csv')
  (options, args) = parser.parse_args()

  main(options.wordVectorFile, options.trainJiaFangDataFile, 
       options.testJiaFangDataFile, options.trainSetModelFile,
       options.testSetModelFile, options.validationSetModelFile,
       options.predictProbFile, options.idRecordFile,
       options.inputFile, options.outputFile, options.groundTruthFile)