import numpy as np
from sklearn import preprocessing

for user_id in range(9):
  mental_filename="./"+str(user_id+1)+"_para/mental_last.csv"
  mental=np.loadtxt(mental_filename,delimiter=",")
  dm=np.diff(mental)

  _dm=preprocessing.minmax_scale(dm)
  norm_dm = _dm*0.8+0.1*np.ones_like(_dm)

  _m=preprocessing.minmax_scale(mental)
  norm_m = _m*0.8+0.1*np.ones_like(_m)

  dm_filename="../data/"+str(user_id+1)+"/dm_calc.csv"
  normalized_mental="../data/"+str(user_id+1)+"/norm_mental.csv"

  np.savetxt(dm_filename,norm_dm,delimiter=",",fmt='%.3f')
  np.savetxt(normalized_mental,norm_m,delimiter=",",fmt='%.3f')
