#coding: utf8
#author: Shuang Zhao (shuang.zhao11@pactera.com)

import optparse
import tensorflow as tf
import numpy as np
import os
from PreProcess import *
from TextCNN import TextCNN

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option("--devSamplePercentage", type=float, default=.3,
                    help='Percentage of the training data used for validation')
  parser.add_option("--trainData", type=str, default='partTrans.data',
                    help='Data source for training.')
  parser.add_option("--embeddingDim", type=int, default=128,
                    help='Dimensionality of character embedding(default: 128)')
  parser.add_option("--kernelSizes", type=str, default='1,1,1,2,3',
                    help='Comma-separated kernel sizes (default: 3,4,5)')
  parser.add_option("--numKernels", type=int, default=128,
                    help='Number of filters per filter size (default: 128)')
  parser.add_option("--dropoutKeepProb", type=float, default=.5,
                    help='Dropout keep probability (default: 0.5)')
  parser.add_option("--l2RegLambda", type=float, default=0.0,
                    help='L2 regularization lambda (default: 0.0)')
  parser.add_option("--numWords", type=int, default=64,
                    help='Number of words kept in each sentence (default: 64)')
  parser.add_option("--batchSize", type=int, default=1024,
                    help='Batch Size (default: 64)')
  parser.add_option("--numEpochs", type=int, default=2,
                    help='Number of training epochs (default: 200)')
  parser.add_option("--evaluateEvery", type=int, default=100,
                    help='Evaluate model on dev every # steps (default: 100)')

  (options, args) = parser.parse_args()

  print("Loading data...")

  data, label, dictLength, wordDict, rawText = \
    openTrainData(options.trainData, lowRate=0, num=options.numWords)

  np.random.seed(10)
  shuffleIndices = np.random.permutation(np.arange(len(label)))
  xShuffled = data[shuffleIndices]
  yShuffled = label[shuffleIndices]
  textShuffled = np.array(rawText)[shuffleIndices]

  # Split train/dev set
  devSampleIndex = -1 * int(options.devSamplePercentage * float(len(yShuffled)))
  xTrain, xDev = xShuffled[:devSampleIndex], xShuffled[devSampleIndex:]
  yTrain, yDev = yShuffled[:devSampleIndex], yShuffled[devSampleIndex:]
  textTrain, textDev = \
    textShuffled[:devSampleIndex], textShuffled[devSampleIndex:]
  del xShuffled, yShuffled, data, label, textShuffled

  sess = tf.Session()
  with sess.as_default():
    cnn = TextCNN(
      sequenceLength = xTrain.shape[1],
      numClasses=yTrain.shape[1],
      vocabSize=5000,
      embeddingSize=options.embeddingDim,
      kernelSizes=list(map(int, options.kernelSizes.split(","))),
      numKernels=options.numKernels,
      l2RegLambda=options.l2RegLambda
    )
    globalStep = tf.Variable(0, name='globalStep', trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    gradsAndVars = optimizer.compute_gradients(cnn.loss)
    trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

    sess.run(tf.global_variables_initializer())

    def trainStep(xBatch, yBatch):
      feedDict = {cnn.inputX: xBatch, cnn.inputY: yBatch,
                  cnn.dropoutKeepProb: options.dropoutKeepProb}
      op, step, loss, accuracy = sess.run(
        [trainOp, globalStep, cnn.loss, cnn.accuracy], feedDict)
      print(f'step {step}, loss {loss}, acc{accuracy}')

    def devStep(xBatch, yBatch):
      feedDict = {cnn.inputX: xBatch, cnn.inputY: yBatch,
                  cnn.dropoutKeepProb: 1}
      step, loss, accuracy = sess.run(
        [globalStep, cnn.loss, cnn.accuracy], feedDict)
      print(f'step {step}, loss {loss}, acc{accuracy}')

    def getPrediction(xBatch, yBatch):
      feedDict = {cnn.inputX: xBatch, cnn.inputY: yBatch,
                  cnn.dropoutKeepProb: 1}
      score, prediction = sess.run([cnn.scores, cnn.predictions], feedDict)
      return score, prediction

    batches = batchIter(list(zip(xTrain, yTrain)),
                        options.batchSize, options.numEpochs)
    for batch in batches:
      xBatch, yBatch = zip(*batch)
      trainStep(xBatch, yBatch)
      currentStep = tf.train.global_step(sess, globalStep)
      if currentStep % options.evaluateEvery == 0:
        print("\nEvaluation:")
        devStep(xDev, yDev)
        print('')
    score, prediction = getPrediction(xDev, yDev)
    outputFile = open('prediction.csv', 'w')
    for i in range(len(prediction)):
      outputFile.write((str(prediction[i]) + ',' + ','.join(textDev[i])) + '\n')
  outputFile.close()