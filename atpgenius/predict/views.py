from django.shortcuts import render_to_response, render, redirect
from django.template import RequestContext
from django.http import HttpResponse
import simplejson as json
from algs.scrape.util import *
from algs.scrape.models import Player, Match
from algs.scrape.crawl import single_scrape
import datetime


# main predict view
def index(request):
  session = mysql_session()
  
  # load stats / info on currently trained model
  # NOTE: TO-DO

  # get all player names
  names, ids = zip(*[(n.name, n.id) for n in session.query(Player).all()])

  # render page
  data = {
    'names' : names,
    'ids' : ids
  }
  return render_to_response('predict/index.html', data, RequestContext(request))


# getting the model's prediction given user's input via AJAX
def get_prediction(request):
  req = request.POST
  session = mysql_session()

  # get the player ids from player names
  p1 = session.query(Player).filter(Player.name == req['p1-name']).first()
  p2 = session.query(Player).filter(Player.name == req['p2-name']).first()
  players = [p1, p2]

  # >> handle unkown players
  for i,p in enumerate(players):
    if p is None:
      print 'Player %s not found.' % (i,)

      # NOTE To-do
      return False

  # make sure player data is up to date - reload if older than 1 day from now
  for i,p in enumerate(players):
    if p.last_updated < datetime.datetime.now() - datetime.timedelta(1):
      single_scrape(p.url)

      # refresh player object
      # ? does sqlalchemy do this automatically?
      players[i] = session.query(Player).filter(Player.name == p.name).first()

  # vectorize the match

  # predict the match winner
  
  # output the results 
  winner = req['p1-name']
  confidence = 65.0
  data = {
    'winner' : winner,
    'confidence' : confidence
  }
  return HttpResponse(json.dumps(data), mimetype="application/json")
