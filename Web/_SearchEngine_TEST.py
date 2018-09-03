#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from Insight_NLP.Web.SearchEngine import SearchEngine

if __name__ == '__main__':
  searcher = SearchEngine()
  result = searcher.search('拼多多', item_nums=100)
  print(result)
