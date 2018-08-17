#coding: utf8
#author: Xinyi Wu (xinyi.wu5@pactera.com)
"""
This modules implements the scraper_reuters which is used to scrap news from
Reuters
"""
import requests
import bs4
import datetime as dt
from crawlers.clean import cleanText
import logging
from NLP import newsTitleFilter, newsSummaryFilter

logger = logging.getLogger("NewsScraper.crawlers.reuters")

def scraperReuters():
  listNews = list()
  url = 'https://www.reuters.com/news/archive/worldNews'
  try:
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    for tag in soup.find_all('div', {'class': 'story-content'}):
      title = cleanText(tag.find('a').find('h3').get_text())
      if not newsTitleFilter(title):
        logger.debug('This news is not in interest based on title: {}'.format(
          title))
        continue
      thisNews = dict()
      thisNews['title'] = title
      thisNews['url'] = 'https://www.reuters.com' + tag.find('a').get('href')
      thisNews['summary'] = cleanText(tag.find('p').get_text())
      if not newsSummaryFilter(thisNews['summary']):
        logger.debug('This news is not in interest based on summary: {}'.format(
          thisNews['summary']))
        continue
      try:
        artResponse = requests.get(thisNews['url'])
        artSoup = bs4.BeautifulSoup(artResponse.content, 'lxml')
        artDate = artSoup.find('div', {'class': 'date_V9eGk'}).get_text()
        thisNews['newsDate'] = str(
          dt.datetime.strptime(artDate[:artDate.rfind(' /')],
                               '%B %d, %Y / %I:%M %p'))
        article = artSoup.find('div', {'class': 'body_1gnLA'})
        texts = [a.get_text() for a in article.find_all('p')]
        text = ''.join(texts)
        thisNews['article'] = cleanText(text)
      except:
        logger.exception("Error in scraping {}".format(thisNews['url']))
      listNews.append(thisNews)
  except:
    logger.exception("Error in scraping {}".format(url))
  return listNews

if __name__ == '__main__':
    pass