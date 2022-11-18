import numpy as np
import pandas as pd

selected_action_mood = pd.read_csv("selected_action.csv",delimiter=",")
selected_action_face= pd.read_csv("selected_action-face.csv",delimiter=",")

#mood
for user_id in range(1,10):
  if(user_id!=4):
    file_name="./"+str(user_id)+"/kibun_test.csv"
    kibun_mood=np.loadtxt(file_name,delimiter=",")

    file_name="./"+str(user_id)+"/kibun_test2.csv"
    kibun_face=np.loadtxt(file_name,delimiter=",")

    _a=selected_action_mood[selected_action_mood['user_num']==user_id]
    action=_a.values[:,0]
    _b=selected_action_face[selected_action_face['user_num']==user_id]
    action_face=_b.values[:,0]
    print(user_id)
    #print(kibun_face)
    #print(action_face)

    mood_kibun_action=np.hstack((np.reshape(kibun_mood,(-1,1)),np.reshape(action,(-1,1))))
    face_kibun_action=np.hstack((np.reshape(kibun_face,(-1,1)),np.reshape(action_face,(-1,1))))
    print(mood_kibun_action)
    #print(face_kibun_action)
