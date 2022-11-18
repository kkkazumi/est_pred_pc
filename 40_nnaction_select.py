import numpy as np
import scipy as sp
from scipy.stats import pearsonr
from sklearn import preprocessing

import optuna
import keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU

FACTOR_NUM = 10
MENTAL_NUM = 1
INPUT_DIM = FACTOR_NUM + MENTAL_NUM
OUTPUT_DIM = 1 #dm
EPOCH = 50000

ACTION_NUM = 5

for name_num in (0,1,2,3,4,5,6,7,8):
  dir_name = "./data/" + str(name_num+1)
  #test to output
  factor_test = np.loadtxt(dir_name + "/factor_test2.csv",delimiter='\t')
  if(name_num!=4):
    _mood = np.loadtxt(dir_name + "/TESTestimated_m2.csv",delimiter=',')

    _m=preprocessing.minmax_scale(_mood)
    mood_test = _m*0.8+0.1*np.ones_like(_m)

    target_row = 2#to 6
    target_len = ACTION_NUM

    model = load_model(dir_name+"/nn_model.h5")

    print(name_num,"user")
    for t in range(1,11):
      if(t<5):
        multi_val = float(t)
        div_val = float(t+1)
      else:
        multi_val = 5.0
        div_val = 5.0

      before_action_count = factor_test[t-1,target_row:target_row+target_len]*multi_val
      print(before_action_count)
      _cand = np.zeros((ACTION_NUM,ACTION_NUM))
      for action in range(ACTION_NUM):
        arr = np.zeros_like(before_action_count)
        arr[action]=1
        _cand[action,:] = before_action_count + arr
      action_cand=_cand/div_val
      _f=factor_test[t-1,:]
      f=np.tile(_f,(ACTION_NUM,1))
      f[:,target_row:target_row+target_len] = action_cand
      m=np.tile(mood_test[t-1],(ACTION_NUM,1))
      xtest=np.hstack((f,m.reshape(-1,1)))

      dm_pred=model.predict(xtest)

      #check 
      a=np.copy(dm_pred)
      order=np.argsort(np.argsort(a.reshape(5))[::-1])+1
      _max_cand_val = dm_pred[np.argmax(a)]
      percentage=a/_max_cand_val*100

      quiz_count=np.ones(ACTION_NUM)*(t)
      action_num=np.linspace(1,ACTION_NUM,ACTION_NUM)

      output_matome=np.hstack((quiz_count.reshape(-1,1),action_num.reshape(-1,1),dm_pred.reshape(-1,1),order.reshape(-1,1),percentage.reshape(-1,1)))
      if(t==1):
        output_all = np.copy(output_matome)
      else:
        output_all=np.append(output_all,output_matome,axis=0)
    #print(output_all)


    #print("input save mode or not")
    #mode = input()
    #if(mode=="save"):
    #  outname = dir_name + "/dm_pred_allactions2.csv"
    #  np.savetxt(outname,output_all,fmt="%.3f")
    #else:
    #  print("not saves")
