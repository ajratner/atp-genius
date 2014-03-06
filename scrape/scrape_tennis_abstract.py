from urllib2 import urlopen


SAMPLE_URL = "http://www.tennisabstract.com/cgi-bin/player.cgi?p=AlexKuznetsov&f=ACareerqqo1"


# functions to save / load page source from file (...FOR TESTING)
def save_url_source(url=SAMPLE_URL, save_file='test1.html'):
  response = urlopen(url)
  source = response.read()
  with open(save_file, 'wb') as f:
    f.write(source)

def load_url_source(save_file='test1.html'):
  with open(save_file. 'rb') as f:
    return f.read()


# function to parse a single tennisabstract player page into match instances + player metadata
def parse_player_page(source):
  pass
  # NOTE: TO-DO

