# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:28:36 2019

@author: osaritac
"""
import numpy as np
from multiprocessing import Process

###run parallel
#def runInParallel(*fns):
#  proc = []
#  for fn in fns:
#    p = Process(target=fn)
#    p.start()
#    proc.append(p)
#  for p in proc:
#    p.join()
#
###function for ranning in parallel
#    R=dict()
#    proc = []
#    for i in day_vector:
#        for j in time_vector:
#            R[i,j]=result(params)
#            args, kwargs = (R[i,j],train_times[j],test_times[j],train_dates[i],test_dates[i],window_range,tunning_range,tunning_range_LP,alpha_range),{"cbar_times":cbar_times[j], "cbar_dates":cbar_dates[i]}
#            p=Process(target=everything, args=args, kwargs=kwargs)
#            p.start()
#            proc.append(p)
#        for p in proc:
#            p.join()
#
#def everything(R,train_times,test_times,train_dates,test_dates,window_range,tunning_range,tunning_range_LP,alpha_range,cbar_times,cbar_dates):
#    R.generate_everything(train_times,test_times,train_dates,test_dates,window_range,tunning_range,tunning_range_LP,alpha_range,cbar_times=cbar_times,cbar_dates=cbar_dates)

#tunning_range=np.concatenate([np.arange(0.80,0.955,0.01),np.arange(0.975,1.05,0.025)])
tunning_range=np.arange(0.75,0.955,0.01)
window_range=np.array([0.001,1/60,6/60,7/60,8/60,9/60,10/60,11/60,12/60,13/60,14/60,15/60,16/60,17/60,18/60,20/60,25/60,0.5,1])
tunning_range_LP=[1]
alpha_range=[1]

params=[{'cd':-j,'mu':i,'time_date':k} for k in ['Monday_From7300_To800','Monday_From17300_To1800','Saturday_From17300_To1800'] for j in [1.5,2,2.5] for i in [2,1,0.5] ]

time1_morning='07:30:00'
time2_morning='07:45:00'

time1_evening='17:30:00'
time2_evening='17:45:00'

cbar_dates={'saturdays':['26/01/2013','02/02/2013','09/02/2013'],'mondays':['28/01/2013','04/02/2013','11/02/2013']}

cbar_times={'mornings':['07:30:00','07:45:00'],'evenings':['17:30:00','17:45:00']}

train_dates={'saturdays':['16/02/2013'],'mondays':['18/02/2013']}
test_dates={'saturdays':['23/02/2013'],'mondays':['25/02/2013']}

train_times={'mornings':[time1_morning,time2_morning],'evenings':[time1_evening,time2_evening]}
test_times={'mornings':[time1_morning,time2_morning],'evenings':[time1_evening,time2_evening]}

day_vector=['saturdays','mondays']
time_vector=['mornings','evenings']


params1=params[0:9] ; params2=params[9:18] ; params3=params[18:27] 
R1=result(params1);R2=result(params2);R3=result(params3)

R1.generate_everything(train_times['mornings'],test_times['mornings'],train_dates['mondays'],test_dates['mondays'],window_range,tunning_range,tunning_range_LP,alpha_range,cbar_times=cbar_times['mornings'],cbar_dates=cbar_dates['mondays'])
R2.generate_everything(train_times['evenings'],test_times['evenings'],train_dates['mondays'],test_dates['mondays'],window_range,tunning_range,tunning_range_LP,alpha_range,cbar_times=cbar_times['evenings'],cbar_dates=cbar_dates['mondays'])
R3.generate_everything(train_times['evenings'],test_times['evenings'],train_dates['saturdays'],test_dates['saturdays'],window_range,tunning_range,tunning_range_LP,alpha_range,cbar_times=cbar_times['evenings'],cbar_dates=cbar_dates['saturdays'])
