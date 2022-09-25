# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 19:44:40 2020

@author: osaritac
"""

import numpy as np
import networkx as nx
import pandas as pd
from functools import reduce
from copy import deepcopy
from gurobipy import *
import plotly.graph_objects as go
#import math
import feather
#import json
import pickle
import glob
from numba import jitclass,jit,njit,prange   # import the decorator
from numba.extending import overload
from numba import int64, float64, boolean    # import the types
from datetime import datetime, date
import os

spec = [
    ('x', float64[:]),
    ('y', float64[:])
]

spec2 = [
    ('x', float64),
    ('y', float64)
]


@jitclass(spec)
class VectorJIT(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def Dist(self,otherP):
        return (np.sqrt( np.power(self.x-otherP.x,2) + np.power(self.y- otherP.y,2)))
    
#@jitclass(spec2)
#class PointJIT(object):
#    def __init__(self,x,y):
#        self.x=x
#        self.y=y
#    def Dist(self,otherP):
#        return (np.sqrt( np.power(self.x-otherP.x,2) + np.power(self.y- otherP.y,2)))

@jit("int64[:](float64[:])", parallel = True, nopython=True)
def unique_return_inverse(vec):
    vec_=np.unique(vec)
    zer=np.zeros(len(vec),dtype=int64)
    ct=0
    for i in vec_:
        zer[np.where(i==np.asarray(vec))]=int64(ct)
        ct+=1
    return(zer)


@jit("int64[:](float64[:])", parallel = True, nopython=True)
def unique_return_index(vec):
    vec_=np.unique(vec)
    zer=np.zeros(len(vec_),dtype=int64)
    ct=0
    for i in vec_:
        zer[ct]=int64(np.where(i==np.asarray(vec))[0][0])
        ct+=1
    return(zer)

@jit("float64(float64, float64,float64[:],float64[:],float64, float64,float64[:],float64[:],float64,float64,float64)", parallel = True, nopython=True)
def c_bar_single_node(Ox1,Oy1,Ox2,Oy2,Dx1,Dy1,Dx2,Dy2,mu,c_d,nb_minutes):
    '''
    Ox -> scalar
    Dx -> vector
    '''
    n = Dx2.shape[0]
    
    ##Assume if one coordinate is exactly the same, then it is the same point!!    
    vec_index=unique_return_index(Ox2)
    vec_inverse=unique_return_inverse(Ox2)
    nn=vec_index.shape[0]
    #Do not recalculate, retrieve later
    PO1 = VectorJIT(Ox1*np.ones(nn),Oy1*np.ones(nn))
    PO2 = VectorJIT(Ox2[vec_index],Oy2[vec_index])
    
    PD1 = VectorJIT(Dx1*np.ones(nn),Dy1*np.ones(nn))
    PD2 = VectorJIT(Dx2[vec_index],Dy2[vec_index])    
    
    mm = np.minimum(PO1.Dist(PD2),PO2.Dist(PD2))
    mm = np.minimum(mm,PO1.Dist(PD1))    
    mm = np.minimum(mm,PO2.Dist(PD1))                         
    
    dd0=PO1.Dist(PO2)+PD1.Dist(PD2) + mm
    dd11=PO1.Dist(PD1) ##same-element-vector
    dd12=PO2.Dist(PD2) 
    dd1=dd11+dd12
    distances=-np.minimum(dd0,dd1)
    
    ##new distance that is more robust
    for i in prange(distances.shape[0]):
        distances[i]=distances[i]*dd11[i]/dd1[i]
    distances=distances[vec_inverse]
    
    #new c_d
    c_d_tilde=c_d+dd11[0]
    
    #print('Distances are: ',distances)
    
    max_d = np.max(distances) # upper bound
    min_d = np.min(distances) # lower bound
    n = Dx2.shape[0] # length of points between bounds
    current_value = mu/(mu + 1.0/nb_minutes*n)*c_d_tilde+1.0/(mu+1.0/nb_minutes*n)*(1/nb_minutes)*np.sum(distances) #current cost
    current_threshold = min_d  #current threshold
    old_g=int64(9999)
    new_g=int64(0)
    
    while (n>1 and np.abs(old_g-new_g)>0.001):
        if current_value - current_threshold>0:
            min_d = current_threshold
        else:
            max_d = current_threshold
        
        old_g= current_value-current_threshold
        #print('maxd: ', max_d)
        #print('mind: ', min_d)    
        current_threshold = (max_d +min_d)/2.0
        l = np.where(distances >= current_threshold)[0]    
        n = l.shape[0]       
        
        #print('old function: ',old_g)
        
        current_value = mu/(mu+1.0/nb_minutes*n)*c_d_tilde +1.0/(mu+1.0/nb_minutes*n)*(1/nb_minutes)*np.sum(distances[l])
        new_g=current_value-current_threshold
        #print('New function: ',new_g)
    
    #distances[l[0]]=-distances[l[0]]    
    return(current_value)

@jit("float64[:](int64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64,float64,float64)", parallel = True, nopython=True)
def c_bar_multiple_node(Cluster,Ox1,Oy1,Ox2,Oy2,Dx1,Dy1,Dx2,Dy2,mu,c_d,nb_minutes):
    dist=np.zeros(np.max(Cluster)+1,dtype=float64)
    ct=np.zeros(np.max(Cluster)+1,dtype=float64)
    ct0=int64(0)
    for i in Cluster:
        #print('Loop: ',i)
        dist[i]=dist[i]+c_bar_single_node(Ox1[ct0],Oy1[ct0],Ox2,Oy2,Dx1[ct0],Dy1[ct0],Dx2,Dy2,float64(mu),float64(c_d),nb_minutes)
        
        ct[i]=ct[i]+1
        #print('Cluster ',i,' ',ct[i],' iteration cbar value is: ',dist[i])
        ct0+=1
    for i in prange(len(dist)):
        #print('Loop: ',i)
        dist[i]=dist[i]/ct[i]
    
    #print('Number of iterations per cluster is: ',ct)
    return(dist)        
            
class simulation(object):

    def __init__(self,params):
           # self.n=params['n_stations']
            self.cd=params['cd']
            self.mu=params['mu']
            self.log=False
            self.cbar_sample=20
            self.save_dist=True
            self.path=params['time_date']
            #self.T=params['T']
    
    def generate_input(self,cbar_dates,cbar_times,cbar_mode='',\
                       dates=['07/01/2013','14/01/2013','21/01/2013','28/01/2013','04/02/2013','11/02/2013','18/02/2013','25/02/2013'],\
                       times=['07:30:00','08:00:00']): # Default: 6 Mondays 7:30-8:00
        
        self.df=feather.read_dataframe('C:/Users/osaritac/Dropbox/Data/ClusteredInput'+self.path+'.feather')    
        self.df = self.df.sort_values(by=['Date','ArrivalTimes'])
        
        self.dates=dates
        self.times=times
        
        ##Encode characteristics for file name in saving the results
        self.hour_2=pd.to_datetime(times[1]).hour
        self.hour_1=pd.to_datetime(times[0]).hour
        
        self.minute_2=pd.to_datetime(times[1]).minute
        self.minute_1=pd.to_datetime(times[0]).minute
        
        self.second_2=pd.to_datetime(times[1]).second
        self.second_1=pd.to_datetime(times[0]).second
        
        self.date_string=reduce(lambda x,y: x +y[0]+y[1]+'_'+y[3]+y[4]+'_',dates,'')
        self.cbar_date_string=reduce(lambda x,y: x +y[0]+y[1]+'_'+y[3]+y[4]+'_',cbar_dates,'')
        ###########################################################
        
        self.type_coord_df=feather.read_dataframe('C:/Users/osaritac/Dropbox/Data/TypeCoordinate'+self.path+'.feather')
        self.type_coord_df = self.type_coord_df.sort_values(by=['ID'])
        
        self.Ox_=np.array(self.df['pickup_longitude'])
        self.Oy_=np.array(self.df['pickup_latitude'])
        self.Dx_=np.array(self.df['dropoff_longitude'])
        self.Dy_=np.array(self.df['dropoff_latitude'])
        
        
        self.initial_cbar(times=cbar_times,dates=cbar_dates) ##uses mu,cd,self.Ox_,unfiltered df
        #self.initial_cbar_unscaled(times=cbar_times,dates=cbar_dates)
        
        self.Ox_type=np.array(self.type_coord_df['pickup_longitude'])
        self.Oy_type=np.array(self.type_coord_df['pickup_latitude'])
        self.Dx_type=np.array(self.type_coord_df['dropoff_longitude'])
        self.Dy_type=np.array(self.type_coord_df['dropoff_latitude'])
        
        #Filter after getting the coordinates 
        self.df=self.df[self.df['Date'].isin(dates)]
        self.df=self.df[(self.df['ArrivalTimes']<=times[1]) & (self.df['ArrivalTimes']>=times[0])]
        
        #Convert to Minutes
        dates=np.unique(self.df['Date'])
        day_order=pd.DataFrame({'Date':dates,'DayOrder':list(range(dates.shape[0]))})
        merged_df=pd.merge(self.df,day_order,on='Date')
        
        days=np.array(merged_df['DayOrder'])
        hours=np.array(pd.to_datetime(self.df['ArrivalTimes']).dt.hour)
        minutes=np.array(pd.to_datetime(self.df['ArrivalTimes']).dt.minute)
        seconds=np.array(pd.to_datetime(self.df['ArrivalTimes']).dt.second)
        
        #Length of time interval for any day
        self.time_interval=np.max(hours*60+minutes+seconds/60)-np.min(hours*60+minutes+seconds/60)
        #Arrival times that are going to be used by the algorithm
        self.df['ArrivalTimesMinutes']=days*self.time_interval+hours*60+minutes+seconds/60

        self.T=self.df.shape[0]
        
        print('Node coordinates are calculated.')
        
        self.lambda_ = np.array(self.df['lambda'])   
        self.dist_ = -self.type_coord_df['self_dist']

        self.MinDist()
        
        batch_count=1
        while os.path.isdir('C:/Users/osaritac/Dropbox/SimulationResults/Batch'+str(batch_count)+'_cd_'+str(self.cd)+'_mu_'+str(self.mu)+self.path):
            batch_count+=1
        
        self.batch_path='C:/Users/osaritac/Dropbox/SimulationResults/Batch'+str(batch_count)+'_cd_'+str(self.cd)+'_mu_'+str(self.mu)+self.path
        os.mkdir(self.batch_path)
        
    def MinDist(self):
        
        
        try:
            #load attributes
            attributes_type_graph = ''
            with open(r'attributes_graph'+self.path+'.txt','r') as f:
                for i in f.readlines():
                    attributes_type_graph=i #string
            attributes_type_graph = eval(attributes_type_graph) # this is original dict with instance dict
            #load edges
            with open('edges_graph'+self.path+'.txt', "rb") as fp:   # Unpickling
                edges_type_graph = pickle.load(fp)

        except:
            print('MinDist file is ...')
            attributes_type_graph=dict()
            edges_type_graph=list()
            #read type pair distances 
            df0=feather.read_dataframe('C:/Users/osaritac/Dropbox/Data/TypePairDistance'+self.path+'.feather')
            for i in range(df0.shape[0]):            
                df=df0.iloc[i,:]
                ID1=int(df['ID1'])
                ID2=int(df['ID2'])
                
                attributes_type_graph[(ID1,ID2)]={'costs':df['c_ij']} 
                edges_type_graph.append((ID1,ID2))
        
            #save attributes
            with open(r'attributes_graph'+self.path+'.txt','w+') as f:
                f.write(str(attributes_type_graph))
            ##save edges
            with open('edges_graph'+self.path+'.txt', "wb") as fp:   #Pickling
                pickle.dump(edges_type_graph, fp, protocol=2)
                
        ##Create the distance graph
        self.G_Orig_type=nx.Graph()
        self.G_Orig_type.add_edges_from(edges_type_graph)
        nx.set_edge_attributes(self.G_Orig_type, attributes_type_graph)    
 
    def init_simulation(self,nbins=100,sampling_parameter=0.1,num_of_parameters=1):

        ##input
        self.arrival_times = np.array(self.df['ArrivalTimesMinutes'])
        ##Spread arrıvals to tıme horizon
        self.arrival_times=np.random.uniform(np.min(self.arrival_times),np.max(self.arrival_times),self.arrival_times.shape[0])
        self.arrival_times=np.sort(self.arrival_times)
        
        self.num_of_parameters=num_of_parameters
        
        #update arrivals/departures
        self.types = np.array(self.df['Cluster']) ##NYC cluster based types
        self.types_2=np.array(self.df['ID']) ### types in the paper
           
        ##calculate if not available in the database         
        self.departure_times = self.arrival_times+np.random.exponential(1/self.mu,self.T)
        self.departure_name = np.argsort(self.departure_times)
        self.departure_times = np.sort(self.departure_times)  
        
        #print('MinDist() is over')

        
    def initial_cbar(self,times,dates):
        date_string=reduce(lambda x,y: x +y[0]+y[1]+'_'+y[3]+y[4]+'_',dates,'')
        
        df_for_cbar=feather.read_dataframe('C:/Users/osaritac/Dropbox/Data/ClusteredInput'+self.path+'.feather')
        df_for_cbar=df_for_cbar[df_for_cbar['Date'].isin(dates)]
        df_for_cbar=df_for_cbar[(df_for_cbar['ArrivalTimes']<=times[1]) & (df_for_cbar['ArrivalTimes']>=times[0])]
        
        hour_2=np.array(pd.to_datetime(times[1]).hour)
        hour_1=np.array(pd.to_datetime(times[0]).hour)
        
        minute_2=np.array(pd.to_datetime(times[1]).minute)
        minute_1=np.array(pd.to_datetime(times[0]).minute)
        
        second_2=np.array(pd.to_datetime(times[1]).second)
        second_1=np.array(pd.to_datetime(times[0]).second)
        
        time_interval=60*(hour_2-hour_1)+\
                      (minute_2-minute_1)+\
                      (second_2-second_1)/60
        
        try: #load cbar
            self.cbar = ''
            with open(r'selfcbar_from_'+str(hour_1)+str(minute_1)+str(second_1)+'_to_'+str(hour_2)+str(minute_2)+str(second_2) +'_dates_' + date_string\
                      +'_mu'+str(self.mu)+'_cd'+str(np.abs(self.cd))+'_trip_neutral.txt','r') as f:
                for i in f.readlines():
                    cbar=i #string
            self.cbar = eval(cbar) # this is original dict with instance dict
        except:
            #Sample the same percentage of nodes from each cluster for c_bar calculation
            print('cbar calculation started.')
            coord=df_for_cbar.groupby('Cluster')[['Cluster','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].apply(lambda x: x.sample(n=self.cbar_sample,replace=True))
            Cluster=int64(coord.iloc[:,0])
            Ox1=float64(coord.iloc[:,1])        
            Oy1=float64(coord.iloc[:,2])      
            Dx1=float64(coord.iloc[:,3])
            Dy1=float64(coord.iloc[:,4])        

            Ox2=float64(self.Ox_)        
            Oy2=float64(self.Oy_)      
            Dx2=float64(self.Dx_)
            Dy2=float64(self.Dy_)
            
            if self.log:
                print('Clusters are: ',Cluster)
                print('Ox1 is: ', Ox1)

            cbar_array=np.array(c_bar_multiple_node(Cluster,Ox1,Oy1,Ox2,Oy2,Dx1,Dy1,Dx2,Dy2,\
                                                          float64(self.mu),float64(self.cd),float64(time_interval*len(dates)))) 
            if self.log:
                print('cbar array is: ', cbar_array)

            self.cbar=dict()

            for i in Cluster:
                self.cbar[i] = cbar_array[i]
            if self.log:
                print('self.cbar array is: ',self.cbar)

            #save cbar
            with open(r'selfcbar_from_'+str(hour_1)+str(minute_1)+str(second_1)+'_to_'+str(hour_2)+str(minute_2)+str(second_2) +'_dates_' + date_string\
                      +'_mu'+str(self.mu)+'_cd'+str(np.abs(self.cd))+'_trip_neutral.txt','w+') as f:
                f.write(str(self.cbar))
               
    def threshold_policy(self,tunning_parameter=1):
        current_nodes = []
        i_arrival = int(0)
        i_departure = int(0)
        cost = 0
        timestep=0
        
        #Initialize the graph
        G_Orig=nx.Graph()
        
        match_list=list()
        matched_nodes=list()
        unmatched_nodes=list()
        penalty_cost=list()
        waiting_time=dict()
        waiting_time_abandoned=list()

        while (i_arrival < self.T) or (i_departure < self.T):
            timestep+=1
            
            if self.log:            
                print('Time Step :',timestep)

            if i_arrival >= self.T:
                is_arrival = False
                break

            elif i_departure >= self.T:
                is_arrival = True
            else:
                is_arrival = (self.arrival_times[i_arrival] < self.departure_times[i_departure])                                
            if is_arrival: 
                time_=self.arrival_times[i_arrival]
                if self.log:                                
                    print('Current nodes: ',current_nodes)
                    print('Arriving node: ',i_arrival)

                ## Look at https://networkx.github.io/documentation/stable/reference/classes/generated/networkx.Graph.update.html
                
                edges = ()
                if len(current_nodes)>0:
                    edges = ((int(i_arrival), int(v),\
                            {'costs':-self.G_Orig_type[self.types_2[i_arrival]][self.types_2[v]]['costs']}) \
                             for v in current_nodes)

                else: #continue because there is only one node: no need to go through the optimization procedure
                    current_nodes.append(int(i_arrival))
                    G_Orig.add_node(i_arrival)                    
                    i_arrival+=1
                    continue

                try:
                    G_Orig.update(edges=edges, nodes=[i_arrival])
                except:
                    G_Orig = G_Orig.copy()
                    G_Orig.update(edges=edges, nodes=[i_arrival])
                    
                #print('G_relevant nodes after:', relevant_nodes)
                relevant_nodes = [ v for v in current_nodes \
                        if np.abs(G_Orig[int(i_arrival)][int(v)]['costs'])<=tunning_parameter*np.abs(self.cbar_unscaled[self.types[v]])]
                relevant_nodes.append(int(i_arrival))
                    
                ##Update current nodes: Notice updating before does not work because we should not match a node with itself
                current_nodes.append(int(i_arrival))
                
                if self.log:                
                    print('G_relevant nodes after:', relevant_nodes)
                
                G_for_opt=G_Orig.subgraph(relevant_nodes).copy()
                G_for_opt.add_edges_from([(i_arrival,'f_n',{'costs': -9999})])
                
                if self.log:                
                    print('Graph for optimization: ',G_for_opt.edges.data())
    
                m = nx.max_weight_matching(G_for_opt, maxcardinality=True, weight='costs')
                    
                #matching i and f_n is fake, do not take it into account after this
                m.discard((i_arrival,'f_n'))
                m.discard(('f_n',i_arrival))
                G_for_opt.remove_edge(i_arrival,'f_n')
                G_for_opt.remove_node('f_n')
                
                if self.log:                   
                    print('m:',m)
                    print('Add to cost: ',sum([i[2] for i in G_for_opt.edge_subgraph(m).edges.data('costs')]))
                    print('cost: ',cost)
                
                ##Remove matched nodes
                recently_matched=reduce(lambda x,y: x + y,m,())
                G_Orig.remove_nodes_from(recently_matched)
                
                cost += sum([i[2] for i in G_for_opt.edge_subgraph(m).edges.data('costs')]) 
                current_nodes = list(set(current_nodes) - set(recently_matched))
                i_arrival += 1
    
                ##Add to matched_nodes
                for new in recently_matched:
                    matched_nodes.append(new)
                    waiting_time[new]=time_-self.arrival_times[new]
                    
                ##Add to matched_edges
                match_list.append(list(m))
                
            else:
                node = self.departure_name[i_departure]
                if node in current_nodes:
                    current_nodes.remove(node)
                    G_Orig.remove_node(node)
                    cost += self.cd+self.dist_[self.types_2[node]]
                    penalty_cost.append(self.cd+self.dist_[self.types_2[node]])
                    waiting_time_abandoned.append(time_-self.arrival_times[node])
                    unmatched_nodes.append(node)
                i_departure += 1
                #update arrivals/departures
        unmatched=set(i for i in range(len(self.departure_times)))-set(matched_nodes)
        try:
            self.total_cost_threshold=np.abs(cost)/(len(unmatched_nodes) + len(matched_nodes))
            percentage_of_unmatched=len(unmatched_nodes)/(len(unmatched_nodes) + len(matched_nodes))
            self.percentage_unmatched_threshold=percentage_of_unmatched
            self.percentage_unmatched_threshold_no_exclusion=len(unmatched)/(len(unmatched_nodes) + len(matched_nodes))
        except:
            self.total_cost_threshold=np.abs(cost)
            self.percentage_unmatched_threshold=999
            self.percentage_unmatched_threshold_no_exclusion=999
        
        self.matched_edges_threshold=match_list
        self.matched_nodes_threshold=matched_nodes
        
        self.waiting_i=np.array([waiting_time[match_list[i][j][0]] for i in range(len(match_list)) for j in range(len(match_list[i]))])
        self.waiting_j=np.array([waiting_time[match_list[i][j][1]] for i in range(len(match_list)) for j in range(len(match_list[i]))])
        self.cost_of_match_threshold=np.array([self.G_Orig_type[self.types_2[match_list[i][j][0]]][self.types_2[match_list[i][j][1]]]['costs'] for i in range(len(match_list)) for j in range(len(match_list[i]))])
        self.cost_i_threshold=np.array([self.G_Orig_type[self.types_2[match_list[i][j][0]]][self.types_2[match_list[i][j][0]]]['costs'] for i in range(len(match_list)) for j in range(len(match_list[i]))])
        self.cost_j_threshold=np.array([self.G_Orig_type[self.types_2[match_list[i][j][1]]][self.types_2[match_list[i][j][1]]]['costs'] for i in range(len(match_list)) for j in range(len(match_list[i]))])
        self.savings_threshold=list(1-self.cost_of_match_threshold/(self.cost_i_threshold+self.cost_j_threshold))
        
        result_frame=pd.DataFrame({'Match':[match_list[i][j] for i in range(len(match_list)) for j in range(len(match_list[i]))],\
                    'Cost of Match':self.cost_of_match_threshold,\
                    'Total Cost':[self.total_cost_threshold]*len(self.savings_threshold),\
                    'Cost1':self.cost_i_threshold,\
                    'Cost2':self.cost_j_threshold,\
                    'Waiting1':self.waiting_i,\
                    'Waiting2':self.waiting_j,\
                    'Savings':self.savings_threshold,\
                    'Savings_Unscaled':(self.cost_i_threshold+self.cost_j_threshold)-self.cost_of_match_threshold,\
                    'Matched':len(matched_nodes)*np.ones(len(self.savings_threshold)),\
                    'Unmatched':len(unmatched_nodes)*np.ones(len(self.savings_threshold)),\
                    'Policy':['Threshold']*len(self.savings_threshold),\
                    'Departure Rate':[self.mu]*len(self.savings_threshold),\
                    'Parameter':[tunning_parameter]*len(self.savings_threshold),\
                    'Test Date':[self.date_string]*len(self.savings_threshold),\
                    'Cbar Date':[self.cbar_date_string]*len(self.savings_threshold),\
                    'Time of Day':[self.path]*len(self.savings_threshold),\
                    'c_d':[self.cd]*len(self.savings_threshold)})
        
        penalty_frame=pd.DataFrame({'PenaltyCost':penalty_cost,\
                                    'WaitingTime':waiting_time_abandoned,\
                                    'Policy':'Threshold',\
                                    'Departure Rate':self.mu,\
                                    'Parameter':tunning_parameter,\
                                    'Test Date':self.date_string,\
                                    'Cbar Date':self.cbar_date_string,\
                                    'Time of Day':self.path,\
                                    'c_d':self.cd})        ##Save result_frame without overwriting
        f_name=self.batch_path+'/ResultFrame_Threshold_tunning_parameter_'+str(tunning_parameter)+'_from_'+str(self.hour_1)+str(self.minute_1)+str(self.second_1)+'_to_'+str(self.hour_2)+str(self.minute_2)+str(self.second_2) +'_dates_' + self.date_string\
                      +'_mu'+str(self.mu)+'_cd'+str(np.abs(self.cd))
        
        file_count=2
        f_present=glob.glob(f_name+'.csv')
        while f_present:
            f_name_final=f_name+'_'+str(file_count)   
            f_present=glob.glob(f_name_final+'.csv')
            file_count+=1
        if file_count==2:
            f_name_final=f_name   
                     
        result_frame.to_csv(f_name_final+'.csv', sep='\t')
        penalty_frame.to_csv(f_name_final+'_penalty.csv',sep='\t')
        
        average_saving=np.mean(self.savings_threshold)
        
        self.average_saving_threshold=average_saving
        return(self.total_cost_threshold)
        
    def Gurobi_LP_policy(self,tunning_parameter=1):
        current_nodes = []
        i_arrival = 0
        i_departure = 0
        cost = 0
        match_list=list()
        matched_nodes=list()
        unmatched_nodes=list()
        penalty_cost=list()
        waiting_time=dict()
        waiting_time_abandoned=list()
        
        G_Orig=nx.Graph()
        while (i_arrival < self.T) or (i_departure < self.T):
            if i_arrival >= self.T:
                is_arrival = False
                break
            elif i_departure >= self.T:
                is_arrival = True
            else:
                is_arrival = (self.arrival_times[i_arrival] < self.departure_times[i_departure])            
            if is_arrival: 
                time_=self.arrival_times[i_arrival]
                if self.log:                
                    print(i_arrival,' arrived')
                
                edges = ()
                if len(current_nodes)>0:
                    edges = ((i_arrival, v,\
                            {'costs':-self.G_Orig_type[self.types_2[i_arrival]][self.types_2[v]]['costs']}) \
                             for v in current_nodes)

                else:
                    current_nodes.append(int(i_arrival))
                    G_Orig.add_node(i_arrival)                    
                    i_arrival+=1
                    continue
                                
                G_Orig.update(edges=edges, nodes=[i_arrival])
                
                #Only now, add the i_arrival because we do not want the edges (i_arrival,i_arrival)
                current_nodes.append(i_arrival)

                #Introduce fake nodes
                mm=max(current_nodes)+1
                G_Orig.add_edges_from([(i,mm+i,{'costs': tunning_parameter*self.cbar[self.types[i]]}) for i in current_nodes])
                
                ####Set up the GUROBI model                
                model = Model('LP Policy - Gurobi')
                model.setParam( 'OutputFlag', False ) #Don't show the Gurobi output.
                edges=[(min(i),max(i)) for i in G_Orig.edges()]
                    
                assign_var=model.addVars(edges,vtype=GRB.BINARY)
                
                model.addConstrs((quicksum(assign_var[i,j] for j in G_Orig.neighbors(i) if j>i if((i,j) in edges))+ \
                                  quicksum(assign_var[j,i] for j in G_Orig.neighbors(i) if j<i if((j,i) in edges))==1) \
                                  for i in current_nodes)

                #We don't have (i,j) and (j,i) in the same graph
                obj_fun=quicksum(assign_var[edge[0],edge[1]]*G_Orig[edge[0]][edge[1]]['costs'] for edge in edges)
                
                model.setObjective(obj_fun,GRB.MAXIMIZE)
                
                model.optimize()
                
                #optimal matching as a set
                m={(u,v) for u,v in assign_var if assign_var[u,v].X==1}                
                #remove fake nodes
                for i in current_nodes:
                    m.discard((i,mm+i))
                    m.discard((mm+i,i))                    
                    G_Orig.remove_edge(i,mm+i)
                    G_Orig.remove_node(mm+i)
                
                cost += sum([i[2] for i in G_Orig.edge_subgraph(m).edges.data('costs')]) #Call the object by the edge, return the total cost
                recently_matched=reduce(lambda x,y: x + y,m,())
                current_nodes = list(set(current_nodes) - set(recently_matched))
                #G_Orig=G_Orig.subgraph(current_nodes).copy()
                #G_Orig=G_Orig.subgraph(current_nodes)
                G_Orig.remove_nodes_from(recently_matched)
                i_arrival += 1
                
                    
                ##Add to matched_nodes
                for new in recently_matched:
                    matched_nodes.append(new)
                    waiting_time[new]=time_-self.arrival_times[new]
                
                ##Add to matched_edges
                match_list.append(list(m))
                
            else:
                node = self.departure_name[i_departure]
                if node in current_nodes:
                    current_nodes.remove(node)
                    G_Orig.remove_node(node)
                    cost += self.cd+self.dist_[self.types_2[node]]
                    penalty_cost.append(self.cd+self.dist_[self.types_2[node]])
                    waiting_time_abandoned.append(time_-self.arrival_times[node])
                    unmatched_nodes.append(node)
                i_departure += 1
        #update arrivals/departures 
        
        unmatched=set(i for i in range(len(self.departure_times)))-set(matched_nodes)
        try:
            self.total_cost_gurobi_LP=np.abs(cost)/(len(unmatched_nodes) + len(matched_nodes))
            percentage_of_unmatched=len(unmatched_nodes)/(len(unmatched_nodes) + len(matched_nodes))
            self.percentage_unmatched_gurobi_LP=percentage_of_unmatched
            self.percentage_unmatched_gurobi_LP_no_exclusion=len(unmatched)/(len(unmatched_nodes) + len(matched_nodes))
        except:
            self.total_cost_gurobi_LP=np.abs(cost)
            self.percentage_unmatched_gurobi_LP=999
            self.percentage_unmatched_gurobi_LP_no_exclusion=999
            
        self.matched_edges_gurobi_LP=match_list
        self.matched_nodes_gurobi_LP=matched_nodes
        
        self.waiting_i=np.array([waiting_time[match_list[i][j][0]] for i in range(len(match_list)) for j in range(len(match_list[i]))])
        self.waiting_j=np.array([waiting_time[match_list[i][j][1]] for i in range(len(match_list)) for j in range(len(match_list[i]))])
        self.cost_of_match_gurobi_LP=np.array([self.G_Orig_type[self.types_2[match_list[i][j][0]]][self.types_2[match_list[i][j][1]]]['costs'] for i in range(len(match_list)) for j in range(len(match_list[i]))])
        self.cost_i_gurobi_LP=np.array([self.G_Orig_type[self.types_2[match_list[i][j][0]]][self.types_2[match_list[i][j][0]]]['costs'] for i in range(len(match_list)) for j in range(len(match_list[i]))])
        self.cost_j_gurobi_LP=np.array([self.G_Orig_type[self.types_2[match_list[i][j][1]]][self.types_2[match_list[i][j][1]]]['costs'] for i in range(len(match_list)) for j in range(len(match_list[i]))])
        self.savings_gurobi_LP=list(1-self.cost_of_match_gurobi_LP/(self.cost_i_gurobi_LP+self.cost_j_gurobi_LP))
        
        result_frame=pd.DataFrame({'Match':[match_list[i][j] for i in range(len(match_list)) for j in range(len(match_list[i]))],\
                    'Cost of Match':self.cost_of_match_gurobi_LP,\
                    'Total Cost':[self.total_cost_gurobi_LP]*len(self.savings_gurobi_LP),\
                    'Cost1':self.cost_i_gurobi_LP,\
                    'Cost2':self.cost_j_gurobi_LP,\
                    'Waiting1':self.waiting_i,\
                    'Waiting2':self.waiting_j,\
                    'Savings':self.savings_gurobi_LP,\
                    'Savings_Unscaled':(self.cost_i_gurobi_LP+self.cost_j_gurobi_LP)-self.cost_of_match_gurobi_LP,\
                    'Matched':len(matched_nodes)*np.ones(len(self.savings_gurobi_LP)),\
                    'Unmatched':len(unmatched_nodes)*np.ones(len(self.savings_gurobi_LP)),\
                    'Policy':['Vertex-Additive']*len(self.savings_gurobi_LP),\
                    'Departure Rate':[self.mu]*len(self.savings_gurobi_LP),\
                    'Parameter':[tunning_parameter]*len(self.savings_gurobi_LP),\
                    'Test Date':[self.date_string]*len(self.savings_gurobi_LP),\
                    'Cbar Date':[self.cbar_date_string]*len(self.savings_gurobi_LP),
                    'Time of Day':[self.path]*len(self.savings_gurobi_LP),\
                    'c_d':[self.cd]*len(self.savings_gurobi_LP)})
        penalty_frame=pd.DataFrame({'PenaltyCost':penalty_cost,\
                                    'WaitingTime':waiting_time_abandoned,\
                                    'Policy':'Vertex-Additive',\
                                    'Departure Rate':self.mu,\
                                    'Parameter':tunning_parameter,\
                                    'Test Date':self.date_string,\
                                    'Cbar Date':self.cbar_date_string,\
                                    'Time of Day':self.path,\
                                    'c_d':self.cd})
        ##Save result_frame without overwriting
        f_name=self.batch_path+'/ResultFrame_Gurobi_LP_tunning_parameter_'+str(tunning_parameter)+'_from_'+str(self.hour_1)+str(self.minute_1)+str(self.second_1)+'_to_'+str(self.hour_2)+str(self.minute_2)+str(self.second_2) +'_dates_' + self.date_string\
                      +'_mu'+str(self.mu)+'_cd'+str(np.abs(self.cd))
        
        file_count=2
        f_present=glob.glob(f_name+'.csv')
        while f_present:
            f_name_final=f_name+'_'+str(file_count)   
            f_present=glob.glob(f_name_final+'.csv')
            file_count+=1
        if file_count==2:
            f_name_final=f_name
                        
        result_frame.to_csv(f_name_final+'.csv', sep='\t')
        penalty_frame.to_csv(f_name_final+'_penalty.csv',sep='\t')

        average_saving=np.mean(self.savings_gurobi_LP)
        
        self.average_saving_gurobi_LP=average_saving
              
        return(self.total_cost_gurobi_LP)
                            
    def Gurobi_batching_policy(self, w = 1.0):

        current_nodes = []
        i_arrival = 0
        i_departure = 0
        time_ = 0
        cost = 0
        match_list=list()
        matched_nodes=list()
        unmatched_nodes = list()
        penalty_cost=list()
        waiting_time_abandoned=list()
        waiting_time=dict()
        G_Orig=nx.Graph()

        while (i_arrival < self.T) or (i_departure < self.T):
            time_p = time_
            if i_arrival >= self.T:
                is_arrival = False
                break

                
            elif i_departure >= self.T:
                is_arrival = True
                time_ = self.arrival_times[i_arrival]
            else:
                is_arrival = (self.arrival_times[i_arrival] < self.departure_times[i_departure])            
                time_ = min(self.arrival_times[i_arrival],self.departure_times[i_departure])

            if (np.floor(time_/w) > np.floor(time_p/w)) and (len(G_Orig.nodes()) > 1): # Solve before moving on to the next batch(if the batch no changes)

#                G_relevant = self.G_Orig.subgraph(current_nodes)
    
                ####Set up the GUROBI model

                model = Model('Batching Policy - Gurobi')
                model.setParam( 'OutputFlag', False )
                model.Params.timeLimit= 10.0
                model.Params.Method = 0

                edges=[(min(i),max(i)) for i in G_Orig.edges()]

                assign_var=model.addVars(edges,vtype=GRB.BINARY, lb =0.0, ub=1.0)
                model.addConstrs((quicksum(assign_var[i,j] for j in G_Orig.neighbors(i) if (j>i) and ((i,j) in edges))+ \
                                  quicksum(assign_var[j,i] for j in G_Orig.neighbors(i) if (j<i) and ((j,i) in edges))<=1) \
                                  for i in current_nodes)

                #We dont have (i,j) and (j,i) in the same graph
                obj_fun=quicksum([assign_var[edge[0],edge[1]]*(1000000 + G_Orig[edge[0]][edge[1]]['costs']) for edge in edges])

                model.setObjective(obj_fun,GRB.MAXIMIZE)


                model.optimize()


                #optimal matching as a set, just a convention in this code

                try:

                    m={(u,v) for u,v in assign_var if assign_var[u,v].X>=0.51}

                    cost += sum([i[2] for i in G_Orig.edge_subgraph(m).edges.data('costs')]) #Return the total cost
                    current_nodes = list(set(current_nodes) - set(reduce(lambda x,y: x + y,m,())))
                    G_Orig = G_Orig.subgraph(current_nodes)
                    
                    ##Add to matched_node
                    for new in reduce(lambda x,y: x + y,m,()):
                        matched_nodes.append(new)
                        waiting_time[new]=time_-self.arrival_times[new]
                    ##Add to matched_edges

                    match_list.append(list(m))
                    
                except:
                    i_arrival += 1

            if is_arrival:

                edges = ()
                nodes = ()                
                if len(current_nodes)>0:
                    edges = [(i_arrival, v, {'costs':-self.G_Orig_type[self.types_2[i_arrival]][self.types_2[v]]['costs']}) for v in G_Orig.nodes]
                else:
                    nodes = [i_arrival]
                try:
                    G_Orig.update(nodes=nodes,edges=edges)           
                except:
                    G_Orig = G_Orig.copy()
                    G_Orig.update(nodes=nodes,edges=edges) 

                current_nodes.append(i_arrival)

                i_arrival += 1

            else:

                node = self.departure_name[i_departure]
                if node in current_nodes:
                    current_nodes.remove(node)
                    G_Orig = G_Orig.subgraph(current_nodes)
                    cost += self.cd+self.dist_[self.types_2[node]]
                    penalty_cost.append(self.cd+self.dist_[self.types_2[node]])
                    waiting_time_abandoned.append(time_-self.arrival_times[node])
                    unmatched_nodes.append(node)

                i_departure += 1

        
        unmatched=set(range(len(self.departure_times)))-set(matched_nodes)
        try:
            self.total_cost_gurobi_batching=np.abs(cost)/(len(unmatched_nodes) + len(matched_nodes))        
            percentage_of_unmatched=(1.0*len(unmatched_nodes))/(len(unmatched_nodes) + len(matched_nodes))
            self.percentage_unmatched_gurobi_batching=percentage_of_unmatched
            self.percentage_unmatched_gurobi_batching_no_exclusion=len(unmatched)/(len(unmatched_nodes) + len(matched_nodes))
        except:
            self.total_cost_gurobi_batching=np.abs(cost)
            self.percentage_unmatched_gurobi_batching=999
            self.percentage_unmatched_gurobi_batching_no_exclusion=999
            
        self.matched_edges_gurobi_batching=match_list
        self.matched_nodes_gurobi_batching=matched_nodes
        
        self.waiting_i=np.array([waiting_time[match_list[i][j][0]] for i in range(len(match_list)) for j in range(len(match_list[i]))])
        self.waiting_j=np.array([waiting_time[match_list[i][j][1]] for i in range(len(match_list)) for j in range(len(match_list[i]))])
        self.cost_of_match_gurobi_batching=np.array([self.G_Orig_type[self.types_2[match_list[i][j][0]]][self.types_2[match_list[i][j][1]]]['costs'] for i in range(len(match_list)) for j in range(len(match_list[i]))])
        self.cost_i_gurobi_batching=np.array([self.G_Orig_type[self.types_2[match_list[i][j][0]]][self.types_2[match_list[i][j][0]]]['costs'] for i in range(len(match_list)) for j in range(len(match_list[i]))])
        self.cost_j_gurobi_batching=np.array([self.G_Orig_type[self.types_2[match_list[i][j][1]]][self.types_2[match_list[i][j][1]]]['costs'] for i in range(len(match_list)) for j in range(len(match_list[i]))])
        self.savings_gurobi_batching=list(1-self.cost_of_match_gurobi_batching/(self.cost_i_gurobi_batching+self.cost_j_gurobi_batching))
        
        result_frame=pd.DataFrame({'Match':[match_list[i][j] for i in range(len(match_list)) for j in range(len(match_list[i]))],\
                    'Cost of Match':self.cost_of_match_gurobi_batching,\
                    'Total Cost':[self.total_cost_gurobi_batching]*len(self.savings_gurobi_batching),\
                    'Cost1':self.cost_i_gurobi_batching,\
                    'Cost2':self.cost_j_gurobi_batching,\
                    'Waiting1':self.waiting_i,\
                    'Waiting2':self.waiting_j,\
                    'Savings':self.savings_gurobi_batching,\
                    'Savings_Unscaled':(self.cost_i_gurobi_batching+self.cost_j_gurobi_batching)-self.cost_of_match_gurobi_batching,\
                    'Matched':len(matched_nodes)*np.ones(len(self.savings_gurobi_batching)),\
                    'Unmatched':len(unmatched_nodes)*np.ones(len(self.savings_gurobi_batching)),\
                    'Policy':['Batching']*len(self.savings_gurobi_batching),\
                    'Departure Rate':[self.mu]*len(self.savings_gurobi_batching),\
                    'Parameter':[w]*len(self.savings_gurobi_batching),\
                    'Test Date':[self.date_string]*len(self.savings_gurobi_batching),\
                    'Cbar Date':[self.cbar_date_string]*len(self.savings_gurobi_batching),\
                    'Time of Day':[self.path]*len(self.savings_gurobi_batching),\
                    'c_d':[self.cd]*len(self.savings_gurobi_batching)})
        penalty_frame=pd.DataFrame({'PenaltyCost':penalty_cost,\
                                    'WaitingTime':waiting_time_abandoned,\
                                    'Policy':'Batching',\
                                    'Departure Rate':self.mu,\
                                    'Parameter':w,\
                                    'Test Date':self.date_string,\
                                    'Cbar Date':self.cbar_date_string,\
                                    'Time of Day':self.path,\
                                    'c_d':self.cd})

        ##Save result_frame without overwriting
        f_name=self.batch_path+'/ResultFrame_Gurobi_Batching_window_'+str(w)+'_from_'+str(self.hour_1)+str(self.minute_1)+str(self.second_1)+'_to_'+str(self.hour_2)+str(self.minute_2)+str(self.second_2) +'_dates_' + self.date_string\
                      +'_mu'+str(self.mu)+'_cd'+str(np.abs(self.cd))
        
        file_count=2
        f_present=glob.glob(f_name+'.csv')
        while f_present:
            f_name_final=f_name+'_'+str(file_count)   
            f_present=glob.glob(f_name_final+'.csv')
            file_count+=1
        if file_count==2:
            f_name_final=f_name 
                      
        result_frame.to_csv(f_name_final+'.csv', sep='\t')
        penalty_frame.to_csv(f_name_final+'_penalty.csv',sep='\t')
        
        average_saving=np.mean(self.savings_gurobi_batching)

        self.average_saving_gurobi_batching=average_saving
       
        return(self.total_cost_gurobi_batching)
        
