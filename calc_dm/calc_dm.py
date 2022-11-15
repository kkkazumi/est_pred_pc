import numpy as np


for user_id in range(9):
  mental_filename="./"+str(user_id+1)+"_para/mental_last.csv"
  mental=np.loadtxt(mental_filename,delimiter=",")
  print(mental)
  dm=np.diff(mental)
  dm_filename="../data/"+str(user_id+1)+"/dm_calc.csv"
  np.savetxt(dm_filename,dm,delimiter=",",fmt='%.3f')
  print(dm)

