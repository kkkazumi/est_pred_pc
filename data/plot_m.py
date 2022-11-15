import numpy as np
import matplotlib.pyplot as plt

for userid in range(9):
  est_filename = "./"+str(userid+1)+"/estm_before.csv"
  ans_filename = "./"+str(userid+1)+"/kibun_before.csv"

  estimated_m = np.loadtxt(est_filename,delimiter=",")
  answered_m = np.loadtxt(ans_filename,delimiter=",")

  fig = plt.figure()
  ax1 = fig.add_subplot(111)

  ax1.plot(estimated_m,linestyle="dashed",marker='o',markerfacecolor="white",color="crimson",label="estimated mood")
  ax1.set_ylabel('estimated mood')
  ax1.set_xlabel('question number')

  ax2=ax1.twinx()
  ax2.plot(answered_m,linestyle="dashed",marker='o',color="limegreen",label="self-asessed mood")
  ax2.set_ylabel('self-asessed mood')
  #ax2.set_ylim()

  ax1_len = max(estimated_m)-min(estimated_m)
  ax2_len = max(answered_m)-min(answered_m)
  ax1.set_ylim(min(estimated_m)-0.2*abs(ax1_len),max(estimated_m)+0.3*abs(ax1_len))
  ax2.set_ylim(min(answered_m)-0.2*abs(ax2_len),max(answered_m)+0.3*abs(ax2_len))

  h1, l1=ax1.get_legend_handles_labels()
  h2, l2=ax2.get_legend_handles_labels()
  ax1.legend(h1+h2,l1+l2)#,loc='upper left', bbox_to_anchor=(1, 1))
  #ax1.legend(h1+h2,l1+l2,loc='upper left', bbox_to_anchor=(1, 1))
  fig_name = "quiz_30_check"+str(userid+1)+".eps"

  #plt.show()
  fig.savefig(fig_name)
  print(userid)
