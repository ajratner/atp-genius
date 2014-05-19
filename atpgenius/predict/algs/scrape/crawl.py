from selenium import webdriver
from scrape_tennis_abstract import parse_player_page
import time
import random
from cPickle import load, dump
import logging

WAIT_MULTIPLIER = 3
SEEN_FILE = 'seen.urls'
QUEUE_FILE = 'queue.urls'

logging.basicConfig(level=logging.ERROR)


# scrape single player page by url
def single_scrape(url):
  driver = webdriver.PhantomJS()
  driver.get(url)
  page_urls = parse_player_page(driver.page_source, url)
  return True


# non-multi-thread crawl routine for *SMALL* sites
def serial_crawl(seed_url):

  # try loading seen & queued urls from saved file
  try:
    new_urls = load(open(QUEUE_FILE, 'rb'))
    seen_urls = load(open(SEEN_FILE, 'rb'))
  except:
    new_urls = [seed_url]
    seen_urls = []

  # instantiate objects
  wait = 0
  driver = webdriver.PhantomJS()
  counter = 0

  # proceed through the urls
  try:
    while len(new_urls) > 0:

      # randomized time delay depending on time to load page
      time.sleep(random.random()*wait*WAIT_MULTIPLIER)
      
      # get page and scrape
      t0 = time.time()
      url = new_urls.pop()
      while url in seen_urls:
        url = new_urls.pop(0)
      print '\n%s -- %s' % (counter, url)
      driver.get(url)
      source = driver.page_source
      wait = time.time() - t0

      # check for unseen urls & update seen list
      new_urls += parse_player_page(source, url)
      
      # mark the url as seen
      seen_urls.append(url)
      counter += 1
    print 'DONE!'

  # handle errors by simply printing for now
  except Exception, e:
    logging.exception("Something awful happened!")

  # save seen and new lists to disk!
  finally:
    print 'Dumping urls to disk...'
    dump(new_urls, open(QUEUE_FILE, 'wb'))
    dump(seen_urls, open(SEEN_FILE, 'wb'))



if __name__ == '__main__':
  serial_crawl('http://www.tennisabstract.com/cgi-bin/player.cgi?p=RafaelNadal&f=ACareerqqo1')
