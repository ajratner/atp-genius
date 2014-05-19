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


# core feature assembly function
# >> vs_p2  =  list of win/losses ~ [1, 0]
# >> lnm = last_n_matches  =  list of (win/loss ~ [1,0], rank-at-time)
def get_match_features(p1_rank, p2_rank, t_round, surface, vs_p2, lnm):
  f = np.zeros(len(N_FEATURES))

  # feature 1 - is player higher rank?
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
  f[4] = float(sum(vs_p2)) / len(vs_p2) if len(vs_p2) > 2 else 0.5

  # feature 6 - winning streak (consecutive previous wins)
  streak = 0
  for m in lnm[::-1]:
    if m[0] < 1:
      break
    streak += 1
  f[5] = float(streak)

  # feature 7 - 5-game moving average change in rank
  f[6] = np.mean([lnm[i][1] - lnm[i-1][1] for i in range(1, len(lnm))])
  
  return f


# vectorize a single (eg unseen) match, based only on certain provided info
def vectorize_new_match(p1, p2, surface, t_round):
  session = mysql_session()

  # get p1,p2 joint match history
  p1w=session.query(Match).filter(and_(Match.p1_name==p1.name, Match.p2_name==p2.name)).count()
  p2w=session.query(Match).filter(and_(Match.p1_name==p2.name, Match.p2_name==p1.name)).count()
  vs_p2 = [1]*len(p1w) + [0]*len(p2w)

  # get last n matches
  last_n = session.query(Match).filter(or_(Match.p1_name==p1.name, Match.p2_name==p1.name)).order_by(desc(Match.date))[:LAST_N_MATCHES]
  lnm = [(1 if m.p1_name==p1.name else 0, m.p1_rank) for m in last_n]

  # get features
  f = get_match_features(p1.rank, p2.rank, t_round, surface, vs_p2, lnm)
  return f

  


  """  
  if match.p1_id == pid:
    vopp = vs_opponent[match.p2_name]
    X[r,4] = float(sum(vopp)) / len(vopp) if len(vopp) > 2 else 0.5
    vs_opponent[match.p2_name].append(1)
  else:
    vopp = vs_opponent[match.p1_name]
    X[r,4] = float(sum(vopp)) / len(vopp) if len(vopp) > 2 else 0.5
    vs_opponent[match.p1_name].append(0)

  # feature 6 - winning streak (consecutive previous wins)
  streak = 0; b = 1
  while r - b > 0 and Y[r-b] == 1:
    streak += 1
    b += 1
  X[r,5] = float(streak)

  # feature 7 - 5-game moving average change in rank
  if r > 5:
    X[r,6] = np.mean([rank_at_time[r-i] - rank_at_time[r-i-1] for i in range(1,5)])
  else:
    X[r,6] = 0.0
  rank_at_time[r] = match.p1_rank if match.p1_id == pid else match.p2_rank
  """
  


# wrapper for vectorizing all matches
def vectorize_matches():
  session = mysql_session()
  pids = [p.id for p in session.query(Player.id).all()]
  xs = []
  ys = []
  for pid in pids:
    x,y = vectorize_player_matches(pid)
    if x is not None:
      xs.append(x)
      ys.append(y)
  X = np.concatenate(xs, axis=0)
  Y = np.concatenate(ys, axis=0)

  # normalize features
  for c in range(N_FEATURES):
    max_abs = max([abs(x) for x in X[:,c]])
    X[:,c] /= max_abs

  # convert X to csr-matrix & return
  return csr_matrix(X), Y

# vectorize all of a certain player's matches
def vectorize_player_matches(pid):
  matches = [m for m in get_matches(pid) if m.p1_rank and m.p2_rank]
  if len(matches) == 0:
    return None, None
  matches.sort(key = lambda m : m.date)
  vs_opponent = defaultdict(list)
  rank_at_time = np.zeros(len(matches))
  X = np.zeros((len(matches), N_FEATURES))
  Y = np.zeros(len(matches))
  for r,match in enumerate(matches):
    pp = 1 if match.p1_id == pid else -1
    Y[r] = pp


  return X, Y


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
    X,Y = vectorize_matches()
    t1 = time(); print 'Vectorized in %s seconds.' % (t1-t0,); t0 = t1

    # train the model and print out accuracy
    print model2(X,Y)
    t1 = time(); print 'Trained in %s seconds.' % (t1-t0,); t0 = t1
