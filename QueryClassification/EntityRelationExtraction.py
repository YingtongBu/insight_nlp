#!/usr/bin/env python
#coding: utf8

#Author: hong xu(hong.xu55@pactera.com)
#todo: code review: no need to mark the modification time.
#Last Modification: 08/16/2018

from itertools import combinations
import pandas as pd
from gensim import corpora, models, similarities
from collections import defaultdict
import numpy as np
import nltk
from fuzzywuzzy import fuzz
import collections

# This is our semi-CFG; Extend it according to your own needs
def defineCFG():
  cfg = {}
  cfg["NNP+NNP"] = "NNP"
  ##cfg["NN+NN"] = "NNP"
  ##cfg["NNP+NN"] = "NNP"
  cfg["JJ+JJ"] = "JJ"
  cfg["JJ+CD"] = "JJ"
  cfg["JJ+NN"] = "NNP"
  cfg["RB+NN"] = "NNP"
  cfg["CD+NN"] = "NNP"
  cfg["VBG+NN"] = "NNP"
  cfg["VBG+DT"] = "VBG"
  return cfg

def eventInput():
  file = open('./BigEvents.data', encoding='utf-8')
  eventDf = pd.DataFrame.from_csv(file,index_col=None, header=None)
  objectDf = eventDf.select_dtypes(['object'])
  eventDf[objectDf.columns] = objectDf.apply(lambda x: x.str.strip())
  
  return eventDf

def makeCompanyDictionary():
  file = open('./S&P500.data', encoding='utf-8')
  df = pd.DataFrame.from_csv(file, index_col=None)
  companyNameList = df["Name"]
  companySymbolList = df["Symbol"]
  companyDictionary = {}
  
  companyTermList = ['(', ')', ',', '.', '.com', 'A', 'C', 'Class', 'Co', 
                   'Co.', 'Communications', 'Companies','Company', 'Company',
                   'Corp', 'Corporation', 'Energy', 'Group', 'Inc', 'Inc.',
                   'Industries', 'Plc', 'Stores', 'Technologies', 
                   'Technology', 'The', 'the', 'com']
  
  for i in range(len(companySymbolList)):
    companyDictionary[companySymbolList[i]] = companyNameList[i]
  
  for company in companyNameList:
    companyDictionary[company] = company
    current = company
    
    companyDictionary[company.lower()] = company
            
    while nltk.word_tokenize(current)[-1] in companyTermList:
      current = current.replace(nltk.word_tokenize(current)[-1],"")
      if current[-1] == " ":
          current = current[:-1]
      companyDictionary[current] = company
      companyDictionary[current.lower()] = company
  
    while nltk.word_tokenize(current)[0] in companyTermList:
      current = current.replace(nltk.word_tokenize(current)[0],"")
      if current[0] == " ":
          current = current[1:]
      companyDictionary[current] = company
      companyDictionary[current.lower()] = company
        
  ##special case for AT&T
  companyDictionary['AT & T'] = 'AT&T Inc.'
  companyDictionary['at & t'] = 'AT&T Inc.'
  companyDictionary['BB & T Corporation'] = 'BB&T Corporation'
  companyDictionary['BB & T'] = 'BB&T Corporation'
  companyDictionary['bb & t Corporation'] = 'BB&T Corporation'
  companyDictionary['bb & t'] = 'BB&T Corporation'
  companyDictionary['Chipotle'] = 'Chipotle Mexican Grill'
  companyDictionary['chipotle'] = 'Chipotle Mexican Grill'
  companyDictionary['Amazon'] = 'Amazon.com Inc.'
  companyDictionary['amazon'] = 'Amazon.com Inc.'
  companyDictionary['McDonalds'] = "McDonald's Corp."
  companyDictionary['Mcdonalds'] = "McDonald's Corp."
  companyDictionary['Google'] = 'Alphabet Inc Class A'
  companyDictionary['Alphabet'] = 'Alphabet Inc Class A'
  companyDictionary['boa'] = 'Bank of America Corp'
  companyDictionary['BOA'] = 'Bank of America Corp'
  companyDictionary['Chase'] = 'JPMorgan Chase & Co.'

  return companyDictionary

def makeSectorDictionary():
  sectorName = ['Industrials', 'Telecommunication Services', 'Financials', 
                'Consumer Discretionary', 'Consumer Staples','Energy', 
                'Materials', 'Real Estate', 'Utilities', 
                'Information Technology', 'Health Care']
  sectorDictionary = {}
  for item in sectorName:
    sectorDictionary[item] = item
    sectorDictionary[item.lower()] = item
  return sectorDictionary

def makeEventDictionary(df):
  ##define three dictionaries
  eventDictionary ={}
  eventLsiDictionary ={}
  eventCorpusDictionary ={}
  
  stoplist = set(['is', 'how'])
  
  ##input the dictionary, corpus, lsi get from each column of 
  ##dataframe into dictionary
  ##add number keys into the dictionary
  i = 0
  for idx, item in enumerate(df.values.tolist()):
    cleanedList = [x for x in item if str(x) != 'nan']
    documents = cleanedList[0:]
    dictionary, lsi, corpus = getLsi(documents, stoplist)
    eventDictionary[item[0]] = dictionary
    eventLsiDictionary[item[0]] = lsi
    eventCorpusDictionary[item[0]] = corpus
    eventDictionary[i] = dictionary
    eventLsiDictionary[i] = lsi
    eventCorpusDictionary[i] = corpus
    idx = idx + 1
  
  return eventDictionary, eventLsiDictionary, eventCorpusDictionary

def getLsi(documents, stoplist):
        
  texts = [[word.lower() for word in document.split()
            if word.lower() not in stoplist]
           for document in documents]
  
  frequency = defaultdict(int)
  for text in texts:
    for token in text:
      frequency[token] += 1
  texts = [[token for token in text if frequency[token] > 1] for text in texts]
  dictionary = corpora.Dictionary(texts)
  # doc2bow counts the number of occurences of each distinct word,
  # converts the word to its integer word id and returns the result
  # as a sparse vector
  
  corpus = [dictionary.doc2bow(text) for text in texts]
  lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
  return dictionary, lsi, corpus

##This is a class for get the entities and company in the sentence
class NPExtractor(object):

  def __init__(self, sentence, companyDic, sectorDic):
      self.sentence = sentence
      self.companyDic = companyDic
      self.sectorDic = sectorDic

  # Split the sentence into singlw words/tokens
  def tokenizeSentence(self, sentence):
      tokens = nltk.word_tokenize(sentence)
      return(tokens)
  
  # Normalize brown corpus' tags ("NN", "NN-PL", "NNS" > "NN")
  def normalizeTags(self, tagged):
    nTagged = []
    for t in tagged:
      if t[1] == "NP-TL" or t[1] == "NP":
        nTagged.append((t[0], "NNP"))
        continue
      if t[1].endswith("-TL"):
        nTagged.append((t[0], t[1][:-3]))
        continue
      if t[1].endswith("S"):
        nTagged.append((t[0], t[1][:-1]))
        continue
      nTagged.append((t[0], t[1]))
    return(nTagged)
  
  def findCompany(self, sentence, dictionary):
    tokens = self.tokenizeSentence(self.sentence)
    ##tags = self.normalize_tags(nltk.pos_tag(tokens))
    
    result = []
    
    for i in range(len(tokens)):
      current = ""
      for j in range(i,len(tokens)):
        if current == "" or tokens[j] in {"'s", "'t", ".", ","}:
          current = current + tokens[j]
        else:
          current = current + " " + tokens[j]
        try:
          company = dictionary[current]
          if company not in result:
            result.append(company)
        except:
          KeyError
                
    return(result)
  
  ##add peoples' names products and organizations and action decection 
  # Extract the main topics from the sentence
  def extract(self):
    cfg = defineCFG()
    tokens = self.tokenizeSentence(self.sentence)
    tags = self.normalizeTags(nltk.pos_tag(tokens))
    
    print(tags)
    
    companyList = self.findCompany(self.sentence, self.companyDic)
    sectorList = self.findCompany(self.sentence, self.sectorDic)
    ##len(company_list) ##!!!!count company
    
    merge = True
    while merge:
      merge = False
      for x in range(0, len(tags)-1):
        t1 = tags[x]
        t2 = tags[x + 1]
        key = "%s+%s" % (t1[1], t2[1])
        value = cfg.get(key, '')
        if value:
          merge = True
          tags.pop(x)
          tags.pop(x)
          match = "%s %s" % (t1[0], t2[0])
          pos = value
          tags.insert(x, (match, pos))
          break

    matches = []
    for t in tags:
        
      if t[1] == "NNP" or t[1] == "NN" or t[1] == "V":
        matches.append(t[0])
    ##duplicate = []
    for item in matches:
      for company in companyList:
        for token in nltk.word_tokenize(company):
          if fuzz.partial_ratio(token, item) == 100:
            matches.remove(item)
            item = item.replace(token, "").strip()
            matches.append(item)
    
    matches = list(filter(None, matches))
    ##print(matches)
    return(matches, companyList, sectorList)

def getAllSubset(wordList):
  s = set(wordList)
  wordSet = sum(map(lambda r: list(combinations(s,r)), range(1,len(s)+1)), [])
  
  newList = []
  for words in wordSet:
    newList.append(' '.join(words)) 

  return newList

def averageSim(sentence, dictionary, lsi, corpus):

  vectorBow = dictionary.doc2bow(sentence.lower().split())
  # convert the query to LSI space
  vectorLsi = lsi[vectorBow]
  index = similarities.MatrixSimilarity(lsi[corpus])
  sims = index[vectorLsi]
  
  return(np.mean(sims))

def getEvent(entityList, companyList, eventDictionaryDic,
              eventLsiDic, eventCorpusDic):
  result = []
  eventList = []
  phraseList = []
  scoreList = []
  
  for phrase in entityList + companyList:
    i = 0
    while i < 100:
      try:
        meanSim = averageSim(phrase, eventDictionaryDic[i], eventLsiDic[i], 
                             eventCorpusDic[i])
        if meanSim > 0.5:
            
          event = (list(eventDictionaryDic.keys())[list(
              eventDictionaryDic.values()).index(eventDictionaryDic[i])])
          result.append(event)
            
        elif meanSim > 0.1:
          event = (list(eventDictionaryDic.keys())[list(
              eventDictionaryDic.values()).index(eventDictionaryDic[i])])
          scoreList.append(meanSim)
          phraseList.append(phrase)
          eventList.append(event)
                
        i = i + 1
      except KeyError:
        break
        
    temp_df = pd.DataFrame(np.column_stack([eventList, phraseList, scoreList]))
    same_event = ([item for item, count in collections.Counter(
        eventList).items() if count > 1])
    
    for item in same_event:
        res = " ".join(temp_df.loc[temp_df[0] == item][1])
        if averageSim(res, eventDictionaryDic[item], eventLsiDic[item], 
                      eventCorpusDic[item])>0.5:
#            print(avg_sim(res, event_dictionary_dic[item], 
#                            event_lsi_dic[item], event_corpus_dic[item]))
          result.append(item)   
    return(list(set(result)))

def classify(sentence, companyDic, sectorDic, eventDictionaryDic, 
             eventLsiDic, eventCorpusDic):
    np_extractor = NPExtractor(sentence, companyDic, sectorDic)
    entityList, companyList, sectorList = np_extractor.extract()
    
    eventList = getEvent(entityList, companyList, eventDictionaryDic, 
                         eventLsiDic, eventCorpusDic)

    print(companyList)
    print(eventList)
    print(sectorList)
    
    if eventList != []:
      eventSet = set([x[0] for x in eventList])
      if eventSet == {'E'}:
        category = 3
        print("3 : influence of events")
      elif eventSet == {'P'}:
        category = 4
        print("4 : influence of policy")
      else:
        category = 5 ##fake
        print("combine of 3 & 4")
    elif len(companyList) > 1 or len(sectorList) > 1:
      category = 2
      print("2 : comparison of companies/industries")
    elif len(companyList) == 1 or len(sectorList) == 1:
      category = 1
      print("1: description of company/industry")
    else:
      category = 0
    return(category, entityList)

if __name__ == '__main__':
  #stoplist = set(['is', 'how'])
  ##inpit dictionary for the companies and sectors
  companyDictionary = makeCompanyDictionary()
  sectorDictionary = makeSectorDictionary()
  
  ##input dataframe for event
  eventDf = eventInput()
  eventDic, eventLsiDic, eventCorpusDic = makeEventDictionary(eventDf)
  
  result = []
  question = "Steve Jobs is the Apple's CEO"
  category, entityList = classify(question, companyDictionary, 
                                  sectorDictionary, eventDic, eventLsiDic, 
                                  eventCorpusDic)
