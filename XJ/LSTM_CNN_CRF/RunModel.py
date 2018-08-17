#!/usr/bin/python
# This scripts loads a pretrained model and a raw .txt files. It then performs sentence splitting and tokenization and passes
# the input sentences to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel.py modelPath inputPath
# For pretrained models see docs/Pretrained_Models.md
from __future__ import print_function
import nltk
from util.preprocessing import addCharInformation, createMatrices, addCasingInformation
from neuralnets.BiLSTM import BiLSTM
import sys

if len(sys.argv) < 3:
    print("Usage: python RunModel.py modelPath inputPath")
    exit()

modelPath = sys.argv[1]
inputPath = sys.argv[2]

# :: Read input ::
with open(inputPath, 'r') as f:
    text = f.read()

# :: Load the model ::
lstmModel = BiLSTM.loadModel(modelPath)


# :: Prepare the input ::
# sentences = [{'tokens': nltk.word_tokenize(sent)} for sent in nltk.sent_tokenize(text)]
text_list = [con for con in text.split('\n') if con != '']
# print(len(text_list))
sentences = [{'tokens': nltk.word_tokenize(words)} for words in text_list]

addCharInformation(sentences)
addCasingInformation(sentences)
dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

# :: Tag the input ::
tags = lstmModel.tagSentences(dataMatrix)

# print(len(tags['Chinese']))

output_file_object = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/output.txt', 'w')
# :: Output to stdout ::
for sentenceIdx in range(len(sentences)):
    tokens = sentences[sentenceIdx]['tokens']
    for tokenIdx in range(len(tokens)):
        tokenTags = []
        for modelName in sorted(tags.keys()):
            tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])
         
        output_file_object.writelines((tokens[tokenIdx] + '\t' + tokenTags[0]))
        output_file_object.write('\n')
    
    output_file_object.write('\n')

output_file_object.close()