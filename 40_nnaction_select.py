import numpy as np
import scipy as sp
from scipy.stats import pearsonr

INPUT_DIM = 11
MID_UNIT = 15
OUTPUT_DIM = 1
EPOCH = 50000

for name_num in (0,1,2,3,4,5,6,7,8):
  dir_name = "./data/" + str(name_num+1)
  #test to output
  factor_test = np.loadtxt(dir_name + "/factor_test.csv",delimiter='\t')
  #TODO get mood
  #mood_test = np.loadtxt(dir_name + "/mood_train" + i_csv ,delimiter=',')

  target_row = 2#to 6j:w
  target_len = 5

  for t in range(1,10):
    if(t<5):
      before_action_count = factor_test[t-1,target_row:target_row+target_len]*t
      for action in range(5):
        arr = np.zeros_like(before_action_count)
        arr[action]=1
        _cand = before_action_count + arr 
        action_cand=_cand/float(t+1)
        print("action cand",action_cand)
    else:
      before_action_count = factor_test[t-1,target_row:target_row+target_len]*5
      for action in range(5):
        arr = np.zeros_like(before_action_count)
        arr[action]=1
        _cand = before_action_count + arr 
        action_cand=_cand/5.0
        print("action cand",action_cand)
    input()

  import optuna
  import keras.backend as K
  from keras.models import Sequential, Model, load_model
  from keras.layers import Input, Dense, Activation
  from keras.layers.advanced_activations import LeakyReLU


  m_pred = np.zeros_like(mood_test)

  model = load_model(dir_name+"/nn_model.h5")

  #set data to test
  for m_t in range(len(mood_test)):
    x_tes = np.tile(factor_test[m_t],(100,1))
    m_candidate = np.linspace(0,1,100)
    y_ans = np.tile(face_test[m_t],(100,1))

    xtest = np.hstack((x_tes,m_candidate.reshape(-1,1)))
    ytest = model.predict(xtest)
    err = ytest - y_ans
    err_array = np.sum(err,1)
    m_est = np.argmin(abs(err_array))
    m_pred[m_t] = m_est/100.0
    print("m_pred",m_pred)

  #test output
  outname = dir_name + "/TRAINestimated_nn" + i_csv
  np.savetxt(outname,m_pred,fmt="%.3f")
