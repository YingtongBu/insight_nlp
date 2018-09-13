#!/usr/bin/env python
#coding: utf8
#author: Xinlu Yu (xinlu.yu1@pactera.com)

import requests
from newspaper import Article
import newspaper
from bs4 import BeautifulSoup

class Scraper(object):
  def __init__(self, url, language='zh'):
    self.url = url
    self._news_article = Article(self.url, language=language)
    self._news_article.download()
    self._news_article.parse()
    self._news_article.nlp()
    self.news_publish_time = self._get_newstime()

  def _get_newstime(self):
    _news_content = requests.get(self.url)
    _news_content.encoding = 'utf-8'
    _news_content = _news_content.text
    _soup = BeautifulSoup(_news_content, 'html.parser')
    _news_time = _soup.find(class_="time")
    _news_time = _news_time.text
    return _news_time

  def get_result(self):
    _news_title = self._news_article.title
    _news_summary = self._news_article.summary
    _news_datetime = self.news_publish_time
    _news_result = {
      'DateTime': _news_datetime,
      'Title': _news_title,
      'Summary': _news_summary,
      'Link': self.url
    }
    return _news_result
