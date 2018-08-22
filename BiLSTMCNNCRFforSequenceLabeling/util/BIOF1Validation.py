#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)
from __future__ import print_function
import logging
def computeF1TokenBasis(predictions, correct, OLabel):  
  prec = computePrecisionTokenBasis(predictions, correct, OLabel)
  rec = computePrecisionTokenBasis(correct, predictions, OLabel)  
  f1 = 0
  if (rec + prec) > 0:
    f1 = 2.0 * prec * rec / (prec + rec)
  return prec, rec, f1

def computePrecisionTokenBasis(guessedSentences, 
                               correctSentences, OLabel):
  assert(len(guessedSentences) == len(correctSentences))
  correctCount = 0
  count = 0
  for sentenceIdx in range(len(guessedSentences)):
    guessed = guessedSentences[sentenceIdx]
    correct = correctSentences[sentenceIdx]
    assert(len(guessed) == len(correct))
    for idx in range(len(guessed)):
      if guessed[idx] != OLabel:
        count += 1
        if guessed[idx] == correct[idx]:
          correctCount += 1
    
  precision = 0
  if count > 0:    
    precision = float(correctCount) / count      
  return precision

def computeF1(predictions, correct, idx2Label, correctBIOErrors='No', 
              encodingScheme='BIO'): 
  labelPred = []    
  for sentence in predictions:
    labelPred.append([idx2Label[element] for element in sentence])
        
  labelCorrect = []    
  for sentence in correct:
    labelCorrect.append([idx2Label[element] for element in sentence])
            
  encodingScheme = encodingScheme.upper()
    
  if encodingScheme == 'IOBES':
    convertIOBEStoBIO(labelPred)
    convertIOBEStoBIO(labelCorrect)                 
  elif encodingScheme == 'IOB':
    convertIOBtoBIO(labelPred)
    convertIOBtoBIO(labelCorrect)
                      
  checkBIOEncoding(labelPred, correctBIOErrors)

  prec = computePrecision(labelPred, labelCorrect)
  rec = computePrecision(labelCorrect, labelPred)
    
  f1 = 0
  if (rec + prec) > 0:
    f1 = 2.0 * prec * rec / (prec + rec)
        
  return prec, rec, f1

def convertIOBtoBIO(dataset):
  for sentence in dataset:
    prevVal = 'O'
    for pos in range(len(sentence)):
      firstChar = sentence[pos][0]
      if firstChar == 'I':
        if prevVal == 'O' or prevVal[1:] != sentence[pos][1:]:
          sentence[pos] = 'B' + sentence[pos][1:] 

      prevVal = sentence[pos]

def convertIOBEStoBIO(dataset):  
  for sentence in dataset:
    for pos in range(len(sentence)):
      firstChar = sentence[pos][0]
      if firstChar == 'S':
        sentence[pos] = 'B' + sentence[pos][1:]
      elif firstChar == 'E':
        sentence[pos] = 'I' + sentence[pos][1:]
                
def computePrecision(guessedSentences, correctSentences):
  assert(len(guessedSentences) == len(correctSentences))
  correctCount = 0
  count = 0  
  for sentenceIdx in range(len(guessedSentences)):
    guessed = guessedSentences[sentenceIdx]
    correct = correctSentences[sentenceIdx]
         
    assert(len(guessed) == len(correct))
    idx = 0
    while idx < len(guessed):
      if guessed[idx][0] == 'B':  
        count += 1
                
        if guessed[idx] == correct[idx]:
          idx += 1
          correctlyFound = True
                    
          while idx < len(guessed) and guessed[idx][0] == 'I':  
            if guessed[idx] != correct[idx]:
              correctlyFound = False
                        
            idx += 1
                    
          if idx < len(guessed):
            if correct[idx][0] == 'I':  
              correctlyFound = False
                        
          if correctlyFound:
            correctCount += 1
        else:
          idx += 1
      else:  
        idx += 1
    
  precision = 0
  if count > 0:    
    precision = float(correctCount) / count
        
  return precision

def checkBIOEncoding(predictions, correctBIOErrors):
  errors = 0
  labels = 0
    
  for sentenceIdx in range(len(predictions)):
    labelStarted = False
    labelClass = None
        
    for labelIdx in range(len(predictions[sentenceIdx])): 
      label = predictions[sentenceIdx][labelIdx]      
      if label.startswith('B-'):
        labels += 1
        labelStarted = True
        labelClass = label[2:]
            
      elif label == 'O':
        labelStarted = False
        labelClass = None
      elif label.startswith('I-'):
        if not labelStarted or label[2:] != labelClass:
          errors += 1        
          if correctBIOErrors.upper() == 'B':
            predictions[sentenceIdx][labelIdx] = 'B-' + label[2:]
            labelStarted = True
            labelClass = label[2:]
          elif correctBIOErrors.upper() == 'O':
            predictions[sentenceIdx][labelIdx] = 'O'
            labelStarted = False
            labelClass = None
          else:
            assert(False)  
            
  if errors > 0:
    labels += errors
    logging.info("Wrong BIO-Encoding %d/%d labels, %.2f%%" 
                 % (errors, labels, errors / float(labels) * 100),)

def testEncodings():
  goldBIO = [['O', 'B-PER', 'I-PER', 'O', 'B-PER', 'B-PER', 'I-PER'],
             ['O', 'B-PER', 'B-LOC', 'I-LOC', 'O', 'B-PER', 'I-PER', 'I-PER'],
             ['B-LOC', 'I-LOC', 'I-LOC', 'B-PER', 'B-PER', 'I-PER', 'I-PER', 
              'O', 'B-LOC', 'B-PER']]

  print("--Test IOBES--")
  goldIOBES = [['O', 'B-PER', 'E-PER', 'O', 'S-PER', 'B-PER', 'E-PER'],
               ['O', 'S-PER', 'B-LOC', 'E-LOC', 'O', 'B-PER', 'I-PER', 
                'E-PER'],
               ['B-LOC', 'I-LOC', 'E-LOC', 'S-PER', 'B-PER', 'I-PER', 'E-PER', 
                'O', 'S-LOC', 'S-PER']]
  convertIOBEStoBIO(goldIOBES)

  for sentenceIdx in range(len(goldBIO)):
    for tokenIdx in range(len(goldBIO[sentenceIdx])):
      assert (goldBIO[sentenceIdx][tokenIdx] == goldIOBES[sentenceIdx]
              [tokenIdx])

  print("--Test IOB--")
  goldIOB = [['O', 'I-PER', 'I-PER', 'O', 'I-PER', 'B-PER', 'I-PER'],
             ['O', 'I-PER', 'I-LOC', 'I-LOC', 'O', 'I-PER', 'I-PER', 'I-PER'],
             ['I-LOC', 'I-LOC', 'I-LOC', 'I-PER', 'B-PER', 'I-PER', 'I-PER', 
              'O', 'I-LOC', 'I-PER']]
  convertIOBtoBIO(goldIOB)

  for sentenceIdx in range(len(goldBIO)):
    for tokenIdx in range(len(goldBIO[sentenceIdx])):
      assert (goldBIO[sentenceIdx][tokenIdx] == 
              goldIOB[sentenceIdx][tokenIdx])

  print("test encodings completed")

if __name__ == "__main__":
  testEncodings()