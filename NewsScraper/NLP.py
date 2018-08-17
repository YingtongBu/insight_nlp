#coding: utf8
# author: Xinyi Wu (xinyi.wu5@pactera.com)
import gensim.summarization

def newsTitleFilter(title):
  return True

def newsSummaryFilter(summary):
  return True

class Processor:
  def __init__(self, news):
    self.listNews = news

  def summarize(self):
    for news in self.listNews:
      news['summaryGensim'] = gensim.summarization.summarize(news['article'],
                                                             word_count=100)

  @property
  def newsList(self):
    return self.listNews

if __name__ == '__main__':
    print(newsSummaryFilter('This is a fake news.'))
    print(newsTitleFilter('Today is a good day.'))