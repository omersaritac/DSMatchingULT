
"""
Created on Wed Nov 20 10:28:36 2019

@author: osaritac
"""

alpha_range=[1]

param=[{'cd':-0.1,'mu':i,'time_date':j } for j in ['Saturday_From7300_To800','Saturday_From17300_To1800','Monday_From7300_To800','Monday_From1100_To11300'] for i in [0.5,1,2]] 

time1_morning='07:30:00'
time2_morning='07:40:00'

time1_midmorning='11:00:00'
time2_midmorning='11:10:00'

time1_evening='17:30:00'
time2_evening='17:40:00'

cbar_dates={'saturdays':['26/01/2013','02/02/2013','09/02/2013','16/02/2013'],'saturdays_insample':['23/02/2013'],'mondays':['28/01/2013','04/02/2013','11/02/2013','18/02/2013'],'mondays_insample':['25/02/2013']}
cbar_times={'mornings':[time1_morning,time2_morning],'midmornings':[time1_midmorning,time2_midmorning],'evenings':[time1_evening,time2_evening]}

train_dates={'saturdays':['16/02/2013'],'saturdays_insample':['23/02/2013'],'mondays':['18/02/2013'],'mondays_insample':['25/02/2013'],'mondays_hard':['04/02/2013','11/02/2013','18/02/2013']}
test_dates={'saturdays':['23/02/2013'],'mondays':['25/02/2013']}

train_times={'mornings':[time1_morning,time2_morning],'midmornings':[time1_midmorning,time2_midmorning],'evenings':[time1_evening,time2_evening]}
test_times={'mornings':[time1_morning,time2_morning],'midmornings':[time1_midmorning,time2_midmorning],'evenings':[time1_evening,time2_evening]}

day_vector=['saturdays','mondays']
time_vector=['mornings','evenings']

for i in range(len(param)):
    S=simulation(param[i])
    if i>5 and i<9:
        S.generate_input(cbar_dates=cbar_dates['mondays_insample'],cbar_times=cbar_times['mornings'],dates=test_dates['mondays'],times=test_times['mornings'])
    elif i>8:    #out of sample cbars
        S.generate_input(cbar_dates=cbar_dates['mondays_insample'],cbar_times=cbar_times['midmornings'],dates=test_dates['mondays'],times=test_times['midmornings'])
    elif i<6 and i>2:
        S.generate_input(cbar_dates=cbar_dates['saturdays_insample'],cbar_times=cbar_times['evenings'],dates=test_dates['saturdays'],times=test_times['evenings'])
    else:
        S.generate_input(cbar_dates=cbar_dates['saturdays_insample'],cbar_times=cbar_times['mornings'],dates=test_dates['saturdays'],times=test_times['mornings'])
        #out of sample cbars
        #S.generate_input(cbar_dates=cbar_dates['saturdays'],cbar_times=cbar_times['mornings'],dates=test_dates['saturdays'],times=test_times['mornings'])   
        
    S.init_simulation()
    
    ct=1
    print('THRESHOLD POLICY')
    for tune in np.arange(0.6,2,0.02):
        for i in range(1):
            S.init_simulation()
            print('Count: ', ct)
            print('Cost: ',S.threshold_policy(tune))
            print('Saving: ',S.average_saving_threshold)
            print('Unmatched: ',S.percentage_unmatched_threshold)
            ct+=1
    
    ct=1
    print('LP POLICY')
    for tune in np.arange(0.7,1.3,0.02):
        for i in range(1):    
            S.init_simulation()
            print('Count: ', ct)
            print('Cost: ',S.Gurobi_LP_policy(tune))
            print('Saving: ',S.average_saving_gurobi_LP)
            print('Unmatched: ',S.percentage_unmatched_gurobi_LP)
            ct+=1
        
    print('Batching-Multiple Run')    
    ct=1
    vec1=np.zeros(10)
    vec2=np.zeros(10)
    vec3=np.zeros(10)
    for win in np.concatenate([[0.00001,0.5/60,1/60,1.5/60,2/60,2.5/60,3/60,3.5/60,4/60],np.arange(5/60,15/60,1/60),np.arange(15/60,30/60,2/60)]):
        for i in range(1):
            S.init_simulation()
            vec1[i]=S.Gurobi_batching_policy(win)
            vec2[i]=S.average_saving_gurobi_batching
            vec3[i]=S.percentage_unmatched_gurobi_batching
        print('Count: ', ct)
        print('Cost: ',np.mean(vec1))
        print('Saving: ',np.mean(vec2))
        print('Unmatched: ',np.mean(vec3))
        ct+=1
    
