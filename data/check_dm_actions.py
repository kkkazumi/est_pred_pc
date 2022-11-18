import numpy as np
import pandas as pd

print("userid,mood,face")
for i in range(1,10):
  if(i!=5):
    datafile="./"+str(i)+"/dm_pred_allactions.csv"
    datafile_face="./"+str(i)+"/dm_pred_allactions2.csv"
    data=pd.read_csv(datafile,delimiter="\s+")
    data_face=pd.read_csv(datafile_face,delimiter="\s+")
    print(i,data['percentage_of_maxdm'].mean(),data_face['percentage_of_maxdm'].mean())
    input()
