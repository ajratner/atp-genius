from urllib2 import urlopen


# functions to save / load page source from file (...FOR TESTING)
def save_url_source(url, save_file='test1.html'):
  response = urlopen(url)
  source = response.read()
  with open(save_file, 'wb') as f:
    f.write(source)


def load_url_source(save_file='test1.html'):
  with open(save_file, 'rb') as f:
    return f.read()
