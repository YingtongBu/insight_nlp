#-*- coding: utf-8 -*-
from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import prepareDataset, loadDatasetPickle
from keras import backend as K

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

datasets = {
    'Chinese': {                                
        'columns': {1: 'tokens', 2: 'NER_BIO'},    
        'label': 'NER_BIO',                     
        'evaluate': True,                       
        'commentSymbol': None}                  
}

embeddingsPath = 'chinese_word_vector.gz'
pickleFile = prepareDataset(embeddingsPath, datasets)
embeddings, mappings, data = loadDatasetPickle(pickleFile)
params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25), 'charEmbeddings': None, 'maxCharLength': 50}

model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.storeResults('results/Chinese_NER_results.csv')
model.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
model.fit(epochs=25)