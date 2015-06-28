from django.shortcuts import render_to_response, render, redirect
from django.template import RequestContext
from django.http import HttpResponse
import simplejson as json
from algs.scrape.util import *
from algs.scrape.models import Player, Match
from algs.scrape.crawl import single_scrape
from algs.predict import vectorize_match 
import datetime
#from sklearn.externals import joblib
from sqlalchemy import or_, and_
import numpy as np
import cPickle


# NOTES:
# http://wiki.bitnami.com/Amazon_cloud/Where_can_I_find_my_AWS_Marketplace_credentials%253f
# >> bitnami application password: lYKdRIPqw3vf
# >> bitnami apache restart: sudo /opt/bitnami/ctlscript.sh restart apache


# given distance of a match vector to the model's decision hyperplane d, and given this same
# data for all the test/train data, compute a prediction 'confidence' as the probability
# that the prediction is correct given it is between d & d+dd from the hyperplane
def confidence_prediction(d, dd=0.05):
  
  # load statistical basis data
  tc_dists = cPickle.load(open('predict/algs/saved_model/tc_dists.pkl', 'rb'))
  tc_scores = cPickle.load(open('predict/algs/saved_model/tc_scores.pkl', 'rb'))

  # compute probability as 'confidence score' and return
  idx = [i for i,tcd in enumerate(tc_dists) if tcd < (d+dd) and tcd >= d]
  return len([i for i in idx if tc_scores[i] == 1]) / float(len(idx))


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

  for p in players:
    print p.name
    print p.atp_rank

  # >> handle unkown players
  for i,p in enumerate(players):
    if p is None:
      print 'Player %s not found.' % (i,)

      # NOTE To-do
      return False

  # make sure player data is up to date - reload if older than 1 day from now
  print '>> checking freshness of player data'
  for i,p in enumerate(players):
    if i==1 or p.last_updated < datetime.datetime.now() - datetime.timedelta(1):
      print '>>> refreshing player %s' % (i,)
      single_scrape(p.url)

      # refresh player object
      # ? does sqlalchemy do this automatically?
      players[i] = session.query(Player).filter(Player.name == p.name).first()

  # vectorize the match
  print '>> vectorizing match'
  f = vectorize_match(players[0].name, players[1].name, players[0].atp_rank, players[1].atp_rank, int(req['tournament-level']), req['surface'], req['tournament-round'], session=session)

  # loading classifier
  print '>> loading classifier'
  clf = cPickle.load(open('predict/algs/saved_model/atp_genius_trained.pkl', 'rb'))

  # >> for RF classifier
  """
  # predict the match winner
  print '>> predicting match winner'
  Y = clf.predict_proba(np.array([f]))
  
  # output the results 
  classes = clf.classes_
  print Y[0]
  print classes
  winner = req['p1-name'] if classes[np.argmax(Y[0])] == 1 else req['p2-name']
  margin = 100.0*(max(Y[0]) - min(Y[0]))
  """

  # >> for Linear SVC classifier
   
  # predict the match winner
  print '>> predicting match winner'
  X = np.array([f])
  Y = clf.predict(X)
  print clf.decision_function(X)
  confidence = 100.0*confidence_prediction(clf.decision_function(X)[0])

  # output the results 
  winner = req['p1-name'] if Y[0] == 1 else req['p2-name']
  data = {
    'winner' : winner,
    'confidence' : '%.2f' % (confidence,),
    'feature-vector' : ['%.3f' % (x,) for x in list(f)],
    'feature-coefficients' : ['%.3f' % (x,) for x in list(clf.coef_[0])]
  }
  return HttpResponse(json.dumps(data), mimetype="application/json")
