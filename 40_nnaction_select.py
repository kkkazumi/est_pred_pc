import optuna

import keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import scipy as sp
from scipy.stats import pearsonr

INPUT_DIM = 11
MID_UNIT = 15
OUTPUT_DIM = 1
EPOCH = 50000

for name_num in (0,1,2,3,4,5,6,7,8):
  dir_name = "./jrm_test/" + str(name_num+1)

  #test to output
  factor_test = np.loadtxt(dir_name + "/factor_train.csv",delimiter='\t')
  mood_test = np.loadtxt(dir_name + "/mood_train" + i_csv ,delimiter=',')

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
