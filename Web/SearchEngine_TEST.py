#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from Insight_NLP.Web.SearchEngine import SearchEngine

if __name__ == '__main__':
  searcher = SearchEngine(key='005808576341306023160:yojc6z7o63u')
  result = searcher.search('拼多多', item_num=10)
  print(result)
