import sys
from scrape.util import *
from scrape.models import Player, Match
import random
import numpy as np
from sqlalchemy import or_, and_, desc
from scipy.sparse import csr_matrix, vstack
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from collections import defaultdict
from time import time


# get the matches from the db
def get_matches(pid=None):
  session = mysql_session()
  if pid:
    return session.query(Match).filter(or_(Match.p1_id == pid, Match.p2_id == pid)).all()
  else:
    return session.query(Match).all()


# Baseline #1- predict winner will be the one with higher rank
# >> 53%
def baseline1(pid=None):
  matches = [m for m in get_matches(pid) if m.p1_rank and m.p2_rank]
  correct = len([m for m in matches if m.p1_rank > m.p2_rank])
  return correct / float(len(matches))


# Baseline #2- random sampling based on the difference between ranks
# >> 68%
def baseline2(pid=None, i=100):
  return np.mean([baseline2_sub(pid) for n in range(i)])

def baseline2_sub(pid=None):
  matches = [m for m in get_matches(pid) if m.p1_rank and m.p2_rank]
  correct = 0
  deltas = [float(m.p1_rank - m.p2_rank) for m in matches]
  min_d = min(deltas)
  max_d = max(deltas)
  d_range = max_d - min_d
  for d in deltas:
    r = random.random()*d_range + min_d
    if r < d:
      correct += 1
  return correct / float(len(matches))


# Model #1- vectorize single-match features only, run through simple ML model
N_FEATURES = 7
LAST_N_MATCHES = 5
UNK_RANK = 1000


# core feature assembly function
# >> vs_p2  =  list of win/losses ~ [1, 0]
# >> lnm = last_n_matches  =  list of (win/loss ~ [1,0], rank-at-time)
def get_match_features(p1_rank, p2_rank, t_round, surface, vs_p2, lnm):
  f = np.zeros(N_FEATURES)

  # feature 1 - is player higher rank?
  # NOTE >> switch this to lower-ranked... this way is confusing!!!
  f[0] = 1 if p1_rank > p2_rank else -1
  
  # feature 2 - rank difference
  f[1] = float(p1_rank - p2_rank)

  # feature 3 - tournament round
  f[2] = float(t_round)

  # feature 4 - clay surface?
  f[3] = 1 if surface == 'Clay' else 0

  # NOTE: to-do: hard surface?

  # NOTE: to-do: grass OR carpet surface?

  # NOTE: difficulty of tournament

  # feature 5 - vs. opponent history
  # f[4] = float(sum(vs_p2)) / len(vs_p2) if len(vs_p2) > 2 else 0.5
  f[4] = vs_p2

  # feature 6 - winning streak (consecutive previous wins)
  streak = 0
  for m in lnm[::-1]:
    if m[0] < 1:
      break
    streak += 1
  f[5] = float(streak)

  # feature 7 - 5-game moving average change in rank
  f[6] = np.mean([lnm[i][1] - lnm[i-1][1] for i in range(1, len(lnm)) if lnm[i][1] and lnm[i-1][1]])
  
  return f


# vectorize a single (eg unseen) match, based only on certain provided info
def vectorize_match(p1_name, p2_name, p1_rank, p2_rank, surface, t_round, session=None):
  session = mysql_session() if not session else session

  # get p1,p2 joint match history
  p1w = session.query(Match).filter(and_(Match.p1_name == p1_name, Match.p2_name == p2_name)).count()
  p2w = session.query(Match).filter(and_(Match.p1_name == p2_name, Match.p2_name == p1_name)).count()
  vs_p2 = float(p1w) / (p1w + p2w) if p1w + p2w > 2 else 0.5

  # get last n matches for p1
  # NOTE: make this p1/p2 symmetric by getting this for p2 also!
  last_n = session.query(Match).filter(or_(Match.p1_name == p1_name, Match.p2_name == p1_name)).order_by(desc(Match.date))[:LAST_N_MATCHES]
  lnm = []
  for m in last_n:
    if m.p1_name == p1_name:
      lnm.append((1, m.p1_rank))
    else:
      lnm.append((0, m.p2_rank))

  # get features
  f = get_match_features(p1_rank, p2_rank, t_round, surface, vs_p2, lnm)
  return f


# NOTE NOTE
# --> need to proceed serially through ALL the matches keeping everything in queue
# --> need custom function...

# vectorize all matches for training / testing
# >> proceed serially through matches by date for speed
def vectorize_data():
  session = mysql_session()
  
  # last N matches as (win/loss, rank at time)
  lnm = defaultdict(list)
  
  # indexed by e.g. 'Bobby Reynolds::Rafael Nadal' w/ alphabetical order
  vs_win = defaultdict(int)
  vs_count = defaultdict(int)

  # go through all the matches by date
  X = []
  Y = []
  for i,m in enumerate(session.query(Match).order_by(Match.date).all()):

    if i%100 == 0:
      print '\t%s' % (i,)

    # vectorize based on current data
    if m.date and m.p1_rank and m.p2_rank:
      vs_key = '::'.join(sorted([m.p1_name, m.p2_name]))
      vs_p2 = vs_win[vs_key] / float(vs_count[vs_key])

      # >> we want to predict whether p1 wins or not.  Since in the data, p1 is by definition
      # >> the winner, we must randomize the ordering here
      if random.random() > 0.5:
        vs_p2 = vs_p2 if ord(m.p1_name[0]) < ord(m.p2_name[0]) else 1.0 - vs_p2
        lnm = lnm[m.p1_name]
        f = get_match_features(m.p1_rank, m.p2_rank, m.tournament_round, m.surface, vs_p2, lnm)
        X.append(f)
        Y.append(1)
      else:
        vs_p2 = vs_p2 if ord(m.p1_name[0]) > ord(m.p2_name[0]) else 1.0 - vs_p2
        lnm = lnm[m.p2_name]
        f = get_match_features(m.p2_rank, m.p1_rank, m.tournament_round, m.surface, vs_p2, lnm)
        X.append(f)
        Y.append(-1)

      # update lnm dict
      lnm[m.p1_name].append((1, m.p1_rank))
      if len(lnm[m.p1_name]) > LAST_N_MATCHES:
        lnm[m.p1_name] = lnm[m.p1_name][1:]
      lnm[m.p2_name].append((0, m.p2_rank))
      if len(lnm[m.p2_name]) > LAST_N_MATCHES:
        lnm[m.p2_name] = lnm[m.p2_name][1:]
      
      # update vs dicts
      vs_win[vs_key] += 1 if ord(m.p1_name[0]) < ord(m.p2_name[0]) else 0
      vs_count[vs_key] += 1

  return np.array(X), np.array(Y)


"""
# vectorize all matches from database
def vectorize_data():
  session = mysql_session()
  player_cache = {}
  X = []
  Y = []

  # go through all the matches in the database
  # NOTE: TO-DO: optimize this with a simple cache?
  for i,m in enumerate(session.query(Match).all()):

    if i%100 == 0:
      print '\t%s' % (i,)

    # >> we want to predict whether p1 wins or not.  Since in the data, p1 is by definition
    # >> the winner, we must randomize the ordering here
    if random.random() > 0.5:
      Y.append(1)
      X.append(vectorize_match(m.p1_name, m.p2_name, m.p1_rank, m.p2_rank, m.surface, m.tournament_round, session=session))
    else:
      Y.append(-1)
      X.append(vectorize_match(m.p2_name, m.p1_name, m.p2_rank, m.p1_rank, m.surface, m.tournament_round, session=session))
  return np.array(X), np.array(Y)
"""


N_FOLDS = 5
def model1(X, Y):
  
  # create stratified K-fold splits
  skf = StratifiedKFold(y=Y, n_folds=N_FOLDS)

  # define the classifier & which hyper-parameters to x-validate over
  clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)
  C_range = 2.0 ** np.arange(-3,10)
  pg = {'C':C_range}
  
  # grid search for best C values in parallel on multiple cores
  grid_clf = GridSearchCV(clf, param_grid=pg, cv=skf, n_jobs=8)
  grid_clf.fit(X,Y)

  # get precision (accuracy)
  clf = grid_clf.best_estimator_
  print clf.coef_
  Y_p = clf.predict(X)
  correct = 0
  for i in range(X.shape[0]):
    if Y[i] == Y_p[i]:
      correct += 1
  return correct / float(X.shape[0])

def model2(X, Y, K=5):

  # >> random forest requires dense arrays...
  X = X.todense()

  # pick classifier
  #clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)
  #clf = SVC()
  clf = RandomForestClassifier(n_estimators=100)

  # randomly pick K folds
  idx = range(X.shape[0])
  random.shuffle(idx)

  # do K-fold testing
  scores = []
  ks = int(len(idx)/float(K))
  for ki in range(K):

    # get the training / testing splits
    idx_kf = idx[ki*ks:(ki+1)*ks]
    idx_train = [0 if i in idx_kf else 1 for i in range(len(idx))]
    idx_test = [1 if i in idx_kf else 0 for i in range(len(idx))]
    X_train = np.compress(idx_train, X, axis=0)
    Y_train = np.compress(idx_train, Y, axis=0)
    X_test = np.compress(idx_test, X, axis=0)
    Y_test = np.compress(idx_test, Y, axis=0)
  
    # train the classifier
    clf.fit(X_train, Y_train)

    # save the trained classifier to disk
    joblib.dump(clf, 'saved_model/atp_genius_trained.pkl')

    # predict and get simple accuracy (precision)
    Y_p = clf.predict(X_test)
    correct = 0
    for i in range(X_test.shape[0]):
      if Y_test[i] == Y_p[i]:
        correct += 1
    scores.append(correct / float(X_test.shape[0]))
  
  # average the results of the k-fold tests
  return np.mean(scores)



if __name__ == '__main__':
  
  if int(sys.argv[1]) == 1:
    print baseline1()

  elif int(sys.argv[1]) == 2:
    print baseline2(i=10)

  elif int(sys.argv[1]) == 3:
    t0 = time()

    # vectorize all the rows
    X,Y = vectorize_data()
    t1 = time(); print 'Vectorized in %s seconds.' % (t1-t0,); t0 = t1

    # train the model and print out accuracy
    print model2(X,Y)
    t1 = time(); print 'Trained in %s seconds.' % (t1-t0,); t0 = t1
