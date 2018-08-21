#!/usr/bin/env python
#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

import os
import optparse
from PreProcessENG import preProcessENG

def mainProcesser(options):
  if options.lanType == 'ENG':
    preProcessENG(options.inputFile)

if __name__ == '__main__':
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option('-i', '--inputFile', default='TotalEvent.txt')
  parser.add_option('-o', '--outputPath', default='result.txt')
  parser.add_option('-l', '--lanType', default='ENG', help='[ENG, CHN]')
  (options, args) = parser.parse_args()
  mainProcesser(options)

  print('--------finished!----------')