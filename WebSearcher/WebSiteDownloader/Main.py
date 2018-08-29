#coding: utf8
#author: Xinyi Wu (xinyi.wu5@pactera.com)
import datetime as dt
import optparse
import logging
import os
import json

from EmailSender import BufferingSMTPHandler
from NLP import Processor
from Crawlers import Scraper

def save_json(list_news, file):
  with open(file + '.json', 'w') as fp:
    json.dump(list_news, fp)

def set_logging(log_path, today, email_addrlist):
  logger = logging.getLogger("NewsScraper")
  logger.setLevel(logging.DEBUG)
  # create the logging file handler
  fh = logging.FileHandler(log_path + '/' + today + '.log')
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %('
                                'message)s')
  fh.setFormatter(formatter)
  # add handler to logger object
  logger.addHandler(fh)
  # add email handler to logger object
  MAILHOST = 'smtp.gmail.com'
  FROM = "pingankensho@gmail.com"
  TO = email_addrlist
  SUBJECT = 'Logging email from NewsScraper'
  logger.addHandler(BufferingSMTPHandler(MAILHOST, FROM, TO, SUBJECT, 1000))
  return logger

def scrape_wsj(scrapers, date_list, news_path):
  # save each day's news in a single file
  for date in date_list:
    scrapers.scraper_wsj(date)
    post_processor = Processor(scrapers.news_list)
    post_processor.summarize()
    save_json(post_processor.news_list, os.path.join(news_path, 'wsj_' +
                                      dt.datetime.strftime(date, '%Y-%m-%d')))

def scrape_reuters(scrapers, news_path):
  scrapers.scraper_reuters()
  post_processor = Processor(scrapers.news_list)
  post_processor.summarize()
  save_json(post_processor.news_list, os.path.join(news_path, 'reuters'))

def scrape_nytimes(scrapers, date_list, news_path):
  scrapers.scraper_nytimes()
  post_processor = Processor(scrapers.news_list)
  post_processor.summarize()
  save_json(post_processor.news_list, os.path.join(news_path, 'nytimes'))
  #nytimes daily newspaper
  for date in date_list:
    scrapers.scraper_nytimesdaily(date)
    post_processor = Processor(scrapers.news_list)
    post_processor.summarize()
    save_json(post_processor.news_list, os.path.join(news_path, 'nytimes_' +
                                      dt.datetime.strftime(date, '%Y-%m-%d')))

if __name__ == '__main__':
  today = dt.date.today().strftime("%Y%m%d")
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option("-e", "--end_date", default=today,
            help="the end DATE of news scraping, in format of 'YYYYmmdd'")
  parser.add_option("-d", "--num_days", type=int, default=1,
            help="NUMBER of days for news scraping")
  parser.add_option("-n", "--news_path", default="news",
            help="save news scraped to FOLDER")
  parser.add_option("-l", "--log_path", default="log",
            help="save log file to FOLDER")
  parser.add_option("-p", "--phantomjs_path",
            default=
            "/Users/Olivia/Downloads/phantomjs-2.1.1-macosx/bin/phantomjs",
            help="change executable path of PhantomJS to PATH")
  parser.add_option("-u", "--wsjusername",
                    default='redoakrichard@gmail.com',
            help="change USERNAME of Wall Street Journal")
  parser.add_option("-w", "--wsjpassword", default='ready2ca',
            help="change PASSWORD of Wall Street Journal")
  parser.add_option("--email_addr", default='xinyi.wu5@pactera.com',
                    help="email ADDRESSES to send logs (separate by ',')")
  parser.add_option("-q", "--quiet",
            action="store_false", dest="verbose", default=True,
            help="don't print status messages to stdout")
  (options, args) = parser.parse_args()

  end_date = dt.datetime.strptime(options.end_date, '%Y%m%d')

  if not os.path.exists(options.news_path):
    os.makedirs(options.news_path)
  if not os.path.exists(options.log_path):
    os.makedirs(options.log_path)

  # Set logging
  logger = set_logging(options.log_path, today, options.email_addr.split())
  logger.info('Started')

  # Start scraping
  scrapers = Scraper(options.phantomjs_path, options.wsjusername,
                     options.wsjpassword)
  date_list = [end_date -
               dt.timedelta(days=x) for x in range(0, options.num_days)]

  scrape_wsj(scrapers, date_list, options.news_path)
  scrape_reuters(scrapers, options.news_path)
  scrape_nytimes(scrapers, date_list, options.news_path)

  scrapers.quit()

  logger.info('Finished')
  logging.shutdown()
