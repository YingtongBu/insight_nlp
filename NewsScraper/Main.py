#coding: utf8
#author: Xinyi Wu (xinyi.wu5@pactera.com)
import datetime as dt
import optparse
import logging
import os
import json

from EmailSender import BufferingSMTPHandler
from NLP import Processor
from crawlers import Scraper

def saveJson(list_news, file):
  with open(file + '.json', 'w') as fp:
    json.dump(list_news, fp)

def setLogging(logPath, today, emailAddrList):
  logger = logging.getLogger("NewsScraper")
  logger.setLevel(logging.DEBUG)
  # create the logging file handler
  fh = logging.FileHandler(logPath + '/' + today + '.log')
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %('
                                'message)s')
  fh.setFormatter(formatter)
  # add handler to logger object
  logger.addHandler(fh)
  # add email handler to logger object
  MAILHOST = 'smtp.gmail.com'
  FROM = "pingankensho@gmail.com"
  TO = emailAddrList
  SUBJECT = 'Logging email from NewsScraper'
  logger.addHandler(BufferingSMTPHandler(MAILHOST, FROM, TO, SUBJECT, 1000))
  return logger

def scrapeWsj(scrapers, dateList, newsPath):
  # save each day's news in a single file
  for date in dateList:
    scrapers.scraperWsj(date)
    postProcessor = Processor(scrapers.newsList)
    postProcessor.summarize()
    saveJson(postProcessor.newsList, os.path.join(newsPath, 'wsj_' +
              dt.datetime.strftime(date, '%Y-%m-%d')))

def scrapeReuters(scrapers, newsPath):
  scrapers.scraperReuters()
  postProcessor = Processor(scrapers.newsList)
  postProcessor.summarize()
  saveJson(postProcessor.newsList, os.path.join(newsPath, 'reuters'))

def scrapeNytimes(scrapers, dateList, newsPath):
  scrapers.scraperNytimes()
  postProcessor = Processor(scrapers.newsList)
  postProcessor.summarize()
  saveJson(postProcessor.newsList, os.path.join(newsPath, 'nytimes'))
  #nytimes daily newspaper
  for date in dateList:
    scrapers.scraperNytimesDaily(date)
    postProcessor = Processor(scrapers.newsList)
    postProcessor.summarize()
    saveJson(postProcessor.newsList, os.path.join(newsPath, 'nytimes_' +
              dt.datetime.strftime(date, '%Y-%m-%d')))

if __name__ == '__main__':
  today = dt.date.today().strftime("%Y%m%d")
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option("-e", "--endDate", default=today,
            help="the end DATE of news scraping, in format of 'YYYYmmdd'")
  parser.add_option("-d", "--numDays", type=int, default=1,
            help="NUMBER of days for news scraping")
  parser.add_option("-n", "--newsPath", default="news",
            help="save news scraped to FOLDER")
  parser.add_option("-l", "--logPath", default="log",
            help="save log file to FOLDER")
  parser.add_option("-p", "--phantomjsPath",
            default=
            "/Users/xinyiwu/Downloads/phantomjs-2.1.1-macosx/bin/phantomjs",
            help="change executable path of PhantomJS to PATH")
  parser.add_option("-u", "--wsjUsername",
                    default='redoakrichard@gmail.com',
            help="change USERNAME of Wall Street Journal")
  parser.add_option("-w", "--wsjPassword", default='ready2ca',
            help="change PASSWORD of Wall Street Journal")
  parser.add_option("-w", "--emailAddr", default='xinyi.wu5@pactera.com',
                    help="email ADDRESSES to send logs (separate by ',')")
  parser.add_option("-q", "--quiet",
            action="store_false", dest="verbose", default=True,
            help="don't print status messages to stdout")
  (options, args) = parser.parse_args()

  endDate = dt.datetime.strptime(options.endDate, '%Y%m%d')

  if not os.path.exists(options.newsPath):
    os.makedirs(options.newsPath)
  if not os.path.exists(options.logPath):
    os.makedirs(options.logPath)

  # Set logging
  logger = setLogging(options.logPath, today, options.emailAddr.split())
  logger.info('Started')

  # Start scraping
  scrapers = Scraper(options.phantomjsPath, options.wsjUsername,
                     options.wsjPassword)
  dateList = [endDate - dt.timedelta(days=x) for x in range(0, options.numDays)]

  scrapeWsj(scrapers, dateList, options.newsPath)
  scrapeReuters(scrapers, options.newsPath)
  scrapeNytimes(scrapers, dateList, options.newsPath)

  scrapers.quit()

  logger.info('Finished')
  logging.shutdown()