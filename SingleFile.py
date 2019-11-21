import numpy as np
import networkx as nx
import pandas as pd
from functools import reduce
from copy import deepcopy
from gurobipy import *
import plotly.graph_objects as go
import math
import feather
import json
import pickle
from numba import jitclass,jit,njit,prange   # import the decorator
from numba.extending import overload
from numba import int64, float64, boolean    # import the types
from datetime import datetime, date

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
    # -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:31:17 2019

@author: osaritac
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:21:05 2019

@author: osaritac
"""
import numpy as np
import networkx as nx
import pandas as pd
from functools import reduce
from copy import deepcopy
from gurobipy import *
import plotly.graph_objects as go
import math
import feather 
from numba import jitclass,jit,njit,prange   # import the decorator
from numba.extending import overload
from numba import int64, float64, boolean    # import the types

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
def c_bar_single_node(Ox1,Oy1,Ox2,Oy2,Dx1,Dy1,Dx2,Dy2,mu,c_d,nb_hours):
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
    dd11=PO1.Dist(PD1) 
    dd12=PO2.Dist(PD2) 
    dd1=dd11+dd12
    distances=-np.minimum(dd0,dd1)
    distances=distances[vec_inverse]
    
    print('Distances are: ',distances)
    
    max_d = np.max(distances) # upper bound
    min_d = np.min(distances) # lower bound
    n = Dx2.shape[0] # length of points between bounds
    current_value = mu/(mu+1.0/nb_hours*n)*c_d +1.0/(mu+1.0/nb_hours*n)*(1/nb_hours)*np.sum(distances) #current cost
    current_threshold = min_d  #current threshold
    old_g=99999
    new_g=0
    #np.abs(old_value-current_value)>0.1
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
        
        current_value = mu/(mu+1.0/nb_hours*n)*c_d +1.0/(mu+1.0/nb_hours*n)*(1/nb_hours)*np.sum(distances[l])
        new_g=current_value-current_threshold
        #print('New function: ',new_g)
    
    #distances[l[0]]=-distances[l[0]]    
    return(current_value)

@jit("float64(float64, float64,float64[:],float64[:],float64, float64,float64[:],float64[:],float64,float64,float64)", parallel = True, nopython=True)
def c_bar_single_node_tilde(Ox1,Oy1,Ox2,Oy2,Dx1,Dy1,Dx2,Dy2,mu,c_d,nb_minutes):
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
    c_d_tilde=c_d*dd11[0]
    
    #print('Distances are: ',distances)
    
    max_d = np.max(distances) # upper bound
    min_d = np.min(distances) # lower bound
    n = Dx2.shape[0] # length of points between bounds
    current_value = mu/(mu + 1.0/nb_minutes*n)*c_d_tilde+1.0/(mu+1.0/nb_minutes*n)*(1/nb_minutes)*np.sum(distances) #current cost
    current_threshold = min_d  #current threshold
    old_g=int64(9999)
    new_g=int64(0)
    #np.abs(old_value-current_value)>0.1
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
def c_bar_multiple_node_tilde(Cluster,Ox1,Oy1,Ox2,Oy2,Dx1,Dy1,Dx2,Dy2,mu,c_d,nb_minutes):
    dist=np.zeros(np.max(Cluster)+1,dtype=float64)
    ct=np.zeros(np.max(Cluster)+1,dtype=float64)
    ct0=int64(0)
    for i in Cluster:
        #print('Loop: ',i)
        dist[i]=dist[i]+c_bar_single_node_tilde(Ox1[ct0],Oy1[ct0],Ox2,Oy2,Dx1[ct0],Dy1[ct0],Dx2,Dy2,float64(mu),float64(c_d),nb_minutes)
        
        ct[i]=ct[i]+1
        #print('Cluster ',i,' ',ct[i],' iteration cbar value is: ',dist[i])
        ct0+=1
    for i in prange(len(dist)):
        #print('Loop: ',i)
        dist[i]=dist[i]/ct[i]
    
    #print('Number of iterations per cluster is: ',ct)
    return(dist)    

@jit("float64[:](int64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64,float64,float64)", parallel = True, nopython=True)
def c_bar_multiple_node(Cluster,Ox1,Oy1,Ox2,Oy2,Dx1,Dy1,Dx2,Dy2,mu,c_d,nb_hours):
    dist=np.zeros(np.max(Cluster)+1,dtype=float64)
    ct=np.zeros(np.max(Cluster)+1,dtype=float64)
    ct0=int64(0)
    for i in Cluster:
        #print('Loop: ',i)
        dist[i]=dist[i]+c_bar_single_node(Ox1[ct0],Oy1[ct0],Ox2,Oy2,Dx1[ct0],Dy1[ct0],Dx2,Dy2,float64(mu),float64(c_d),nb_hours)
        
        ct[i]=ct[i]+1
        print('Cluster ',i,' ',ct[i],' iteration cbar value is: ',dist[i])
        ct0+=1
    for i in prange(len(dist)):
        #print('Loop: ',i)
        dist[i]=dist[i]/ct[i]
    
    print('Number of iterations per cluster is: ',ct)
    return(dist)    

def tables(simul):
    result_cost_table=go.Table(header=dict(values=['Mu','c_d','Setting','Batching Cost','LP Cost','Threshold Cost']),\
                                                     cells=dict(values=[[simul[i].mu for i in [0.2,0.5,1,0.2,0.5,1]],\
                                                                    [np.abs(simul[i].cd) for i in [0.2,0.5,1,0.2,0.5,1]],\
                                                                    [1,1,1,2,2,2],\
                                                                    [round(simul[i].total_cost_batching[j],2) for j in [0,1] for i in [0.2,0.5,1]],\
                                                                    [round(simul[i].total_cost_gurobi_LP[j],2) for j in [0,1] for i in [0.2,0.5,1]],\
                                                                    [round(simul[i].total_cost_threshold[j],2) for j in [0,1] for i in [0.2,0.5,1]]]))
        
    result_performance_measure_table=go.Table(header=dict(values=['Mu','c_d','Setting',\
                                                    'Batching Saving','LP Saving','Thresh. Saving',\
                                                    'Batching Unmatch','LP Unmatch','Thresh. Unmatch']),\
                                                     cells=dict(values=[[simul[i].mu for i in [0.2,0.5,1,0.2,0.5,1]],\
                                                                    [np.abs(simul[i].cd) for i in [0.2,0.5,1,0.2,0.5,1]],\
                                                                    [1,1,1,2,2,2],\
                                                                    [str(round(100*simul[i].average_saving_batching[j],1))+'%' for j in [0,1] for i in [0.2,0.5,1]],\
                                                                    [str(round(100*simul[i].average_saving_gurobi_LP[j],1))+'%' for j in [0,1] for i in [0.2,0.5,1]],\
                                                                    [str(round(100*simul[i].average_saving_threshold[j],1))+'%' for j in [0,1] for i in [0.2,0.5,1]],\
                                                                    [str(round(100*simul[i].percentage_unmatched_batching[j],1))+'%' for j in [0,1] for i in [0.2,0.5,1]],\
                                                                    [str(round(100*simul[i].percentage_unmatched_gurobi_LP[j],1))+'%' for j in [0,1] for i in [0.2,0.5,1]],\
                                                                    [str(round(100*simul[i].percentage_unmatched_threshold[j],1))+'%' for j in [0,1] for i in [0.2,0.5,1]]]))
    go.Figure(data=[result_cost_table]).show
    go.Figure(data=[result_performance_measure_table]).show
    
    go.Figure(data=[result_cost_table]).write_image('Result_Cost.jpg')
    go.Figure(data=[result_performance_measure_table]).write_image('Result_Performance_Measure.jpg')
class Point(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def Dist(self,otherP):
        return (-np.sqrt( np.power(self.x-otherP.x,2) + np.power(self.y- otherP.y,2)))
            
    
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
    
    def generate_input(self,cbar_dates,cbar_times,\
                       dates=['07/01/2013','14/01/2013','21/01/2013','28/01/2013','04/02/2013','11/02/2013','18/02/2013','25/02/2013'],\
                       times=['07:30:00','08:00:00']): # Default: 6 Mondays 7:30-8:00
        
        self.df=feather.read_dataframe('C:/Users/osaritac/Documents/Research/Data/ClusteredInput'+self.path+'.feather')    
        
        
        self.dates=dates
        self.times=times
        
        self.type_coord_df=feather.read_dataframe('C:/Users/osaritac/Documents/Research/Data/TypeCoordinate'+self.path+'.feather')
        self.Ox_=np.array(self.df['pickup_longitude'])
        self.Oy_=np.array(self.df['pickup_latitude'])
        self.Dx_=np.array(self.df['dropoff_longitude'])
        self.Dy_=np.array(self.df['dropoff_latitude'])
        
        self.initial_cbar(times=cbar_times,dates=cbar_dates) ##uses mu,cd,self.Ox_,unfiltered df
        
        self.Ox_type=np.array(self.type_coord_df['pickup_longitude'])
        self.Oy_type=np.array(self.type_coord_df['pickup_latitude'])
        self.Dx_type=np.array(self.type_coord_df['dropoff_longitude'])
        self.Dy_type=np.array(self.type_coord_df['dropoff_latitude'])
        
        #Filter after getting the coordinates 
        self.df=self.df[self.df['Date'].isin(dates)]
        self.df=self.df[(self.df['ArrivalTimes']<=times[1]) & (self.df['ArrivalTimes']>=times[0])]
        
        self.df = self.df.sort_values(by=['Date','ArrivalTimes'])        
        
        
        #Convert to Minutes
        dates=np.unique(self.df['Date'])
        day_order=pd.DataFrame({'Date':dates,'DayOrder':list(range(dates.shape[0]))})
        merged_df=pd.merge(self.df,day_order,on='Date')
        
        days=merged_df['DayOrder']
        hours=pd.to_datetime(self.df['ArrivalTimes']).dt.hour
        minutes=pd.to_datetime(self.df['ArrivalTimes']).dt.minute
        seconds=pd.to_datetime(self.df['ArrivalTimes']).dt.second
        
        #Length of time interval for any day
        self.time_interval=max(hours*60+minutes+seconds/60)-min(hours*60+minutes+seconds/60)
        #Arrival times that are going to be used by the algorithm
        self.df['ArrivalTimesMinutes']=days*self.time_interval+hours*60+minutes+seconds/60

        self.T=self.df.shape[0]
        
        print('Node coordinates are calculated.')
        
        self.lambda_ = np.array(self.df['lambda'])   
        self.dist_ = -np.sqrt( np.power(self.Ox_type - self.Dx_type,2) + np.power(self.Oy_type - self.Dy_type,2))

    def find_relevant_edges(self):
        temp_frame=pd.DataFrame({'Arrival':self.arrival_times,'Departure':self.departure_times,\
                                 'No':[i for i in range(len(self.departure_times))]})
        V_t=dict()
        ct=0
        for i in self.arrival_times:           
            #Node no of relevant nodes
            print('Arrival time is: ',i)
            active=temp_frame[[x and y for x,y in zip(list(temp_frame['Arrival']<=i),list(temp_frame['Departure']>=i))]]['No']
            print('Active at',i,': ',active)
            V_t[ct]=np.array(list(set(active)))
            ct+=1
        return(V_t)
        
    def det_alphalist_oneplusepsilon(self,nbins):  
        a_dict=dict()
        for node in self.G_Orig.nodes:
            aa=[self.G_Orig[node][n]['costs'] for n in self.G_Orig.neighbors(node)] #array of neighboring nodes for node
            mm=min(aa) #we use it twice
            rr=max(aa)/mm
            epsilon=rr**(1/nbins)
            a_dict[node]=[mm*(epsilon**i) for i in range(nbins+1)] 
        return a_dict
        
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
            df0=feather.read_dataframe('C:/Users/osaritac/Documents/Research/Data/TypePairDistance'+self.path+'.feather')
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
                
        ##Create the type graph (coordinate type, not the cbar type)
        self.G_Orig_type=nx.Graph()
        self.G_Orig_type.add_edges_from(edges_type_graph)
        nx.set_edge_attributes(self.G_Orig_type, attributes_type_graph)    
 
    def init_simulation(self,nbins=100,sampling_parameter=0.1,num_of_parameters=1):
        
        ##input
        self.arrival_times = np.array(self.df['ArrivalTimesMinutes'])

        self.num_of_parameters=num_of_parameters
        
        #update arrivals/departures
        self.types = np.array(self.df['Cluster'])
        self.types_2=np.array(self.df['ID']) 
           
        ##calculate if not available in the database         
        self.departure_times = self.arrival_times+np.random.exponential(1/self.mu,self.T)
        self.departure_name = np.argsort(self.departure_times)
        self.departure_times = np.sort(self.departure_times) 
        
        self.MinDist()
        
        print('MinDist() is over')

        
    def initial_cbar(self,times,dates):
        date_string=reduce(lambda x,y: x +y[0]+y[1]+'_'+y[3]+y[4]+'_',dates,'')
        df_for_cbar=feather.read_dataframe('C:/Users/osaritac/Documents/Research/Data/ClusteredInput'+self.path+'.feather')
        df_for_cbar=df_for_cbar[df_for_cbar['Date'].isin(dates)]
        df_for_cbar=df_for_cbar[(df_for_cbar['ArrivalTimes']<=times[1]) & (df_for_cbar['ArrivalTimes']>=times[0])]
        
        hour_2=pd.to_datetime(times[1]).hour
        hour_1=pd.to_datetime(times[0]).hour
        
        minute_2=pd.to_datetime(times[1]).minute
        minute_1=pd.to_datetime(times[0]).minute
        
        second_2=pd.to_datetime(times[1]).second
        second_1=pd.to_datetime(times[0]).second
        
        time_interval=60*(hour_2-hour_1)+\
                      (minute_2-minute_1)+\
                      (second_2-second_1)/60
        
        try: #load cbar
            self.cbar = ''
            with open(r'selfcbar_from_'+str(hour_1)+str(minute_1)+str(second_1)+'_to_'+str(hour_2)+str(minute_2)+str(second_2) +'_dates_' + date_string\
                      +'_mu'+str(self.mu)+'_cd'+str(np.abs(self.cd))+'_multiplicative.txt','r') as f:
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

            cbar_array=np.array(c_bar_multiple_node_tilde(Cluster,Ox1,Oy1,Ox2,Oy2,Dx1,Dy1,Dx2,Dy2,\
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
                      +'_mu'+str(self.mu)+'_cd'+str(np.abs(self.cd))+'_multiplicative.txt','w+') as f:
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
                        if np.abs(G_Orig[int(i_arrival)][int(v)]['costs'])<=tunning_parameter*np.abs(self.cbar[self.types[v]])]
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
                
                cost += sum([i[2] for i in G_for_opt.edge_subgraph(m).edges.data('costs')]) #Return the total cost
                current_nodes = list(set(current_nodes) - set(recently_matched))
                i_arrival += 1
    
            ##Add to matched_nodes
                for new in recently_matched:
                    matched_nodes.append(new)
                    
                ##Add to matched_edges
                match_list.append(list(m))
                
            else:
                node = self.departure_name[i_departure]
                if node in current_nodes:
                    current_nodes.remove(node)
                    G_Orig.remove_node(node)
                    cost += self.cd*self.dist_[self.types_2[node]]
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
        
        result_frame=pd.DataFrame({'Match':[match_list[i][j] for i in range(len(match_list)) for j in range(len(match_list[i]))],\
                    'Cost of Match':[-self.G_Orig_type[self.types_2[match_list[i][j][0]]][self.types_2[match_list[i][j][1]]]['costs'] for i in range(len(match_list)) for j in range(len(match_list[i]))],\
                    'Cost1':[self.dist_[self.types_2[match_list[i][j][0]]] for i in range(len(match_list)) for j in range(len(match_list[i]))],\
                    'Cost2':[self.dist_[self.types_2[match_list[i][j][1]]] for i in range(len(match_list)) for j in range(len(match_list[i]))]})
        savings=list(1-result_frame['Cost of Match']/(result_frame['Cost1']+result_frame['Cost2']))
        average_saving=np.mean(savings)
        
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
                
                ##Add to matched_edges
                match_list.append(list(m))
                
            else:
                node = self.departure_name[i_departure]
                if node in current_nodes:
                    current_nodes.remove(node)
                    G_Orig.remove_node(node)
                    cost += self.cd*self.dist_[self.types_2[node]]
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
        
        result_frame=pd.DataFrame({'Match':[match_list[i][j] for i in range(len(match_list)) for j in range(len(match_list[i]))],\
                    'Cost of Match':[-self.G_Orig_type[self.types_2[match_list[i][j][0]]][self.types_2[match_list[i][j][1]]]['costs'] for i in range(len(match_list)) for j in range(len(match_list[i]))],\
                    'Cost1':[self.dist_[self.types_2[match_list[i][j][0]]] for i in range(len(match_list)) for j in range(len(match_list[i]))],\
                    'Cost2':[self.dist_[self.types_2[match_list[i][j][1]]] for i in range(len(match_list)) for j in range(len(match_list[i]))]})
        savings=list(1-result_frame['Cost of Match']/(result_frame['Cost1']+result_frame['Cost2']))
        average_saving=np.mean(savings)
        
        self.average_saving_gurobi_LP=average_saving
              
        return(self.total_cost_gurobi_LP)
        
    def Hybrid_policy(self, alpha, w = 1.0):
        current_nodes = []
        i_arrival = 0
        i_departure = 0
        time_ = 0
        cost = 0
        match_list=list()
        matched_nodes=list()
        unmatched_nodes=list()
        G_Orig=nx.Graph()
        
        while (i_arrival < self.T) or (i_departure < self.T):
            time_p = time_
            if i_arrival >= self.T:
                is_arrival = False
                time_ = self.departure_times[i_departure]
                break
            elif i_departure >= self.T:
                is_arrival = True
                time_ = self.arrival_times[i_arrival]
                
            else:
                is_arrival = (self.arrival_times[i_arrival] < self.departure_times[i_departure])            
                time_ = min(self.arrival_times[i_arrival],self.departure_times[i_departure])
               
            #print('time_ is ', time_)
            #print('time_p is ', time_p)
            if np.floor(time_/w) > np.floor(time_p/w): # Solve before moving on to the next batch(if the batch no changes)
                
                #Introduce fake nodes
                if len(current_nodes)>0:
                    mm=max(current_nodes)+1
                else:
                    continue
                #print('Self cbar vector is: ', self.cbar)
                #print('Self types vector is: ', self.types)
                G_Orig.add_edges_from([(i,mm+i,{'costs': alpha*self.cbar[self.types[i]]}) for i in current_nodes])                
                
                ####Set up the GUROBI model
                
                model = Model('Hybrid Policy - Gurobi')
                model.setParam( 'OutputFlag', False ) #Don't show the Gurobi output.
                edges=[(min(i),max(i)) for i in G_Orig.edges()]
                    
                assign_var=model.addVars(edges,vtype=GRB.BINARY)
                
                print('Current Nodes:', current_nodes)
                model.addConstrs((quicksum(assign_var[i,j] for j in G_Orig.neighbors(i) if j>i if((i,j) in edges))+ \
                                  quicksum(assign_var[j,i] for j in G_Orig.neighbors(i) if j<i if((j,i) in edges))==1) \
                                  for i in current_nodes)

                #We don't have (i,j) and (j,i) in the same graph
                obj_fun=quicksum(assign_var[edge[0],edge[1]]*G_Orig[edge[0]][edge[1]]['costs'] for edge in edges)
                
                model.setObjective(obj_fun,GRB.MAXIMIZE)
                
                model.optimize()                
                
                #optimal matching as a set, just a convention in this code
                
                m={(u,v) for u,v in assign_var if assign_var[u,v].X==1}
                print('Match: ',m)
                
                for i in current_nodes:
                    m.discard((i,mm+i))
                    m.discard((mm+i,i))                    
                    G_Orig.remove_edge(i,mm+i)
                    G_Orig.remove_node(mm+i)
                
                cost += sum([i[2] for i in G_Orig.edge_subgraph(m).edges.data('costs')]) #Return the total cost
                print('Cost: ',cost)
                current_nodes = list(set(current_nodes) - set(reduce(lambda x,y: x + y,m,())))
                G_Orig.subgraph(current_nodes).copy()
                
                ##Add to matched_nodes
                for new in reduce(lambda x,y: x + y,m,()):
                    matched_nodes.append(new)
                
                ##Add to matched_edges
                match_list.append(list(m))
                
                print('Nodes in matching: ',set(reduce(lambda x,y: x + y,m,())))
                print('Current Nodes after match: ', current_nodes)
                
            if is_arrival:
                edges = ()
                if len(current_nodes)>0:
                    edges = ((int(i_arrival), int(v),\
                            {'costs':-self.G_Orig_type[self.types_2[i_arrival]][self.types_2[v]]['costs']}) \
                             for v in current_nodes)

                else:
                    current_nodes.append(int(i_arrival))
                    G_Orig.add_node(i_arrival)                    
                    i_arrival+=1
                    continue

                G_Orig.update(edges=edges, nodes=[i_arrival])
                print('Edges are: ',G_Orig.edges())
                print('Nodes are: ',G_Orig.nodes())
                    
                current_nodes.append(i_arrival)
                i_arrival += 1
            else:
                node = self.departure_name[i_departure]
                if node in current_nodes:
                    current_nodes.remove(node)
                    G_Orig.remove_node(node)
                    cost += self.cd*self.dist_[self.types_2[node]]
                    unmatched_nodes.append(node)
                i_departure += 1
          
                
        unmatched=set(range(len(self.departure_times)))-set(matched_nodes)        
        
        try:
            self.total_cost_hybrid=np.abs(cost)/(len(unmatched_nodes) + len(matched_nodes))
            percentage_of_unmatched=len(unmatched_nodes)/(len(unmatched_nodes) + len(matched_nodes))
            self.percentage_unmatched_hybrid=percentage_of_unmatched
            self.percentage_unmatched_hybrid_no_exclusion=len(unmatched)/(len(unmatched_nodes) + len(matched_nodes))
        except:
            self.total_cost_hybrid=np.abs(cost)
            self.percentage_unmatched_hybrid=999
            self.percentage_unmatched_hybrid_no_exclusion=999
        
        self.matched_edges_hybrid=match_list
        self.matched_nodes_hybrid=matched_nodes
        
        result_frame=pd.DataFrame({'Match':[match_list[i][j] for i in range(len(match_list)) for j in range(len(match_list[i]))],\
                    'Cost of Match':[-self.G_Orig_type[self.types_2[match_list[i][j][0]]][self.types_2[match_list[i][j][1]]]['costs'] for i in range(len(match_list)) for j in range(len(match_list[i]))],\
                    'Cost1':[self.dist_[self.types_2[match_list[i][j][0]]] for i in range(len(match_list)) for j in range(len(match_list[i]))],\
                    'Cost2':[self.dist_[self.types_2[match_list[i][j][1]]] for i in range(len(match_list)) for j in range(len(match_list[i]))]})
        savings=list(1-result_frame['Cost of Match']/(result_frame['Cost1']+result_frame['Cost2']))
        average_saving=np.mean(savings)
        
        self.average_saving_hybrid=average_saving
                
    def Gurobi_batching_policy(self, w = 1.0):

        current_nodes = []
        i_arrival = 0
        i_departure = 0
        time_ = 0
        cost = 0
        match_list=list()
        matched_nodes=list()
        unmatched_nodes = list()
        G_Orig=nx.Graph()

        while (i_arrival < self.T) or (i_departure < self.T):
            time_p = time_
            if i_arrival >= self.T:
                is_arrival = False
                break

                time_ = self.departure_times[i_departure]

                model = Model('Batching Policy - Gurobi')
                model.setParam( 'OutputFlag', False )
                model.Params.timeLimit= 100.0
                model.Params.Method = 0
                
                edges=[(min(i),max(i)) for i in G_Orig.edges()]

                assign_var=model.addVars(edges,vtype=GRB.BINARY, lb =0.0, ub=1.0)

                model.addConstrs((quicksum(assign_var[i,j] for j in G_Orig.neighbors(i) if (j>i) and ((i,j) in edges))+ \
                                  quicksum(assign_var[j,i] for j in G_Orig.neighbors(i) if (j<i) and ((j,i) in edges))<=1) \
                                  for i in current_nodes)

                obj_fun=quicksum([assign_var[edge[0],edge[1]]*(10000000 + G_Orig[edge[0]][edge[1]]['costs']) for edge in edges])

                model.setObjective(obj_fun,GRB.MAXIMIZE)

                model.optimize()

                #optimal matching as a set, just a convention in this code

                try:
                    m={(u,v) for u,v in assign_var if assign_var[u,v].X>=0.51}
                    cost += sum([i[2] for i in G_Orig.edge_subgraph(m).edges.data('costs')]) #Return the total cost
                    current_nodes = list(set(current_nodes) - set(reduce(lambda x,y: x + y,m,())))
                    G_Orig = G_Orig.subgraph(current_nodes)
                    
                    ##Add to matched_nodes

                    for new in reduce(lambda x,y: x + y,m,()):

                        matched_nodes.append(new)

                    ##Add to matched_edges

                    match_list.append(list(m))
                except:
                    i_arrival += 1
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
                    
                    ##Add to matched_nodes

                    for new in reduce(lambda x,y: x + y,m,()):
                        matched_nodes.append(new)

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
                    cost += self.cd*self.dist_[self.types_2[node]]
                    unmatched_nodes.append(node)

                i_departure += 1
            
        #update arrivals/departures
        
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

        result_frame=pd.DataFrame({'Match':[match_list[i][j] for i in range(len(match_list)) for j in range(len(match_list[i]))],\
                    'Cost of Match':[-self.G_Orig_type[self.types_2[match_list[i][j][0]]][self.types_2[match_list[i][j][1]]]['costs'] for i in range(len(match_list)) for j in range(len(match_list[i]))],\
                    'Cost1':[self.dist_[self.types_2[match_list[i][j][0]]] for i in range(len(match_list)) for j in range(len(match_list[i]))],\
                    'Cost2':[self.dist_[self.types_2[match_list[i][j][1]]] for i in range(len(match_list)) for j in range(len(match_list[i]))]})

        savings=list(1-result_frame['Cost of Match']/(result_frame['Cost1']+result_frame['Cost2']))
        average_saving=np.mean(savings)

        self.average_saving_gurobi_batching=average_saving
       
        return(self.total_cost_gurobi_batching)
        
    def parameter_selection(self,\
                            times,\
                            dates,\
                            cbar_times,\
                            cbar_dates,\
                            window_range=np.array([0.001,0.01,0.1,6/60,8/60,10/60,0.5]),\
                            tunning_range=np.arange(0.95,1.05,0.01),tunning_range_LP=[1],alpha_range=np.arange(0,1.5,0.75),\
                            hybrid=False):
        
        virt_params={'cd':self.cd,'mu':self.mu,'time_date':self.path}
        virt_self=simulation(virt_params)
        virt_self.generate_input(times=times,dates=dates,cbar_times=cbar_times,cbar_dates=cbar_dates)
        virt_self.init_simulation()
        
        cost_list_batching=[]
        cost_list_hybrid=[]
        
        for win in window_range:
            #l is 3; a parameter for lambda calculation
            #lambda is 3/n on average; window parameter should scale with that
            virt_self.Gurobi_batching_policy(w=win)
            cost_list_batching.append(virt_self.total_cost_gurobi_batching)
            #print('Current window range is ', win)
            if hybrid:
                for alpha in alpha_range:
                    virt_self.Hybrid_policy(w=win,alpha=alpha)    
                    cost_list_hybrid.append(virt_self.total_cost_hybrid)
        
        self.cost_list_batching=cost_list_batching
        ##obtain the optimal window parameter for batching
        self.optimal_window_par=window_range[np.argmin(cost_list_batching)]
        ##run batching with the optimal window parameter
        self.Gurobi_batching_policy(w=self.optimal_window_par)
        
        ##run hybrid with the optimal parameter
        if hybrid:
            min_index=np.argmin(cost_list_hybrid)
            optimal_window_index=np.floor(min_index/len(alpha_range))
            optimal_alpha_index=min_index-optimal_window_index*len(alpha_range)
        
            self.optimal_window_hybrid=window_range[int(optimal_window_index)]
            self.optimal_alpha_hybrid=alpha_range[int(optimal_alpha_index)]
            self.Hybrid_policy(w=self.optimal_window_hybrid,alpha=self.optimal_alpha_hybrid)
        else:
            self.optimal_window_hybrid=999
            self.optimal_alpha_hybrid=999
        
        #Tunning for threshold policy
        cost_list=[]
        for tune in tunning_range:
            virt_self.threshold_policy(tunning_parameter=tune)
            cost_list.append(virt_self.total_cost_threshold)
        
        self.cost_list_threshold=cost_list
        self.optimal_tunning_parameter=tunning_range[np.argmin(cost_list)]
        
        ##run threshold with the optimal parameter
        self.threshold_policy(tunning_parameter=self.optimal_tunning_parameter)
        
        ##Tunning for LP_policy
        cost_list=[]
        for tune in tunning_range_LP:
            virt_self.Gurobi_LP_policy(tunning_parameter=tune)
            cost_list.append(virt_self.total_cost_gurobi_LP)
        
        self.cost_list_LP=cost_list
        self.optimal_tunning_parameter_LP=tunning_range_LP[np.argmin(cost_list)]
        
        ##run the LP policy with the optimal parameter
        self.Gurobi_LP_policy(tunning_parameter=self.optimal_tunning_parameter)


class result(object):
    
    def __init__(self,params): #this should be a set of parameter dictionaries
        self.params=params
        
    def generate_results(self,train_times,test_times,train_dates,test_dates,window_range,tunning_range,tunning_range_LP,alpha_range,cbar_times,cbar_dates,hybrid=False):
        ll=len(self.params)
        self.simul=[simulation(self.params[i]) for i in range(ll)]
        
        #initialize attribute

        self.hybrid_cost=[0 for i in range(ll)]
        self.hybrid_saving=[0 for i in range(ll)]
        self.hybrid_unmatched=[0 for i in range(ll)]
        
        self.thresh_cost=[0 for i in range(ll)]
        self.batching_cost=[0 for i in range(ll)]
        self.LP_cost=[0 for i in range(ll)]
        
        
        self.thresh_saving=[0 for i in range(ll)]
        self.batching_saving=[0 for i in range(ll)]
        self.LP_saving=[0 for i in range(ll)]
        
        
        self.thresh_unmatched=[0 for i in range(ll)]
        self.batching_unmatched=[0 for i in range(ll)]
        self.LP_unmatched=[0 for i in range(ll)]
        
        
        for i in range(len(self.params)):
            self.simul[i].generate_input(times=test_times,dates=test_dates,cbar_times=cbar_times,cbar_dates=cbar_dates)
            self.simul[i].init_simulation()
            
            self.simul[i].parameter_selection(window_range=window_range,tunning_range=tunning_range,tunning_range_LP=tunning_range_LP\
                      ,alpha_range=alpha_range,times=train_times,dates=train_dates,cbar_times=cbar_times,cbar_dates=cbar_dates)
            
            if hybrid:
                self.hybrid_cost[i]=round(self.simul[i].total_cost_hybrid,2)    
                self.hybrid_saving[i]=round(self.simul[i].average_saving_hybrid,2)
                self.hybrid_unmatched[i]=round(self.simul[i].percentage_unmatched_hybrid,2)
                
            self.thresh_cost[i]=round(self.simul[i].total_cost_threshold,4)
            self.batching_cost[i]=round(self.simul[i].total_cost_gurobi_batching,4)
            self.LP_cost[i]=round(self.simul[i].total_cost_gurobi_LP,4)
            
            self.thresh_saving[i]=round(self.simul[i].average_saving_threshold,4)
            self.batching_saving[i]=round(self.simul[i].average_saving_gurobi_batching,4)
            self.LP_saving[i]=round(self.simul[i].average_saving_gurobi_LP,4)
            
            self.thresh_unmatched[i]=round(self.simul[i].percentage_unmatched_threshold,4)
            self.batching_unmatched[i]=round(self.simul[i].percentage_unmatched_gurobi_batching,4)
            self.LP_unmatched[i]=round(self.simul[i].percentage_unmatched_gurobi_LP,4)
            
    def generate_table(self):
        ll=len(self.params)
        self.result_cost_table=go.Table(header=dict(values=['Mu',\
                                                    'c_d','Batching Policy','LP Policy','Threshold Policy','Hybrid Policy']),\
                                                     cells=dict(values=[[self.simul[i].mu for i in range(ll)],\
                                                                    [np.abs(self.simul[i].cd) for i in range(ll)],\
                                                                    [self.batching_cost[i] for i in range(ll)],\
                                                                    [self.LP_cost[i] for i in range(ll)],\
                                                                    [self.thresh_cost[i] for i in range(ll)],\
                                                                    [self.hybrid_cost[i] for i in range(ll)]]))
        
        self.result_performance_measure_table=go.Table(header=dict(values=['Mu',\
                                                    'c_d',\
                                                    'Batching Saving','LP Saving','Threshold Saving','Hybrid Saving',\
                                                    'Batching Unmatched','LP Unmatched','Threshold Unmatched','Hybrid Unmatched']),\
                                                     cells=dict(values=[[self.simul[i].mu for i in range(ll)],\
                                                                    [np.abs(self.simul[i].cd) for i in range(ll)],\
                                                                    self.batching_saving,\
                                                                    self.LP_saving,\
                                                                    self.thresh_saving,\
                                                                    self.hybrid_saving,\
                                                                    self.batching_unmatched,\
                                                                    self.LP_unmatched,\
                                                                    self.thresh_unmatched,\
                                                                    self.hybrid_unmatched]))
        
        self.result_parameter_table=go.Table(header=dict(values=['Alpha for Hybrid','Window for Hybrid','Window for Batching','Tunning par. Threshold','Tunning par. LP']),\
                                                     cells=dict(values=[[self.simul[i].optimal_alpha_hybrid for i in range(ll)],\
                                                                    [self.simul[i].optimal_window_hybrid*60 for i in range(ll)],\
                                                                    [self.simul[i].optimal_window_par*60 for i in range(ll)],\
                                                                    [self.simul[i].optimal_tunning_parameter for i in range(ll)],\
                                                                    [self.simul[i].optimal_tunning_parameter_LP for i in range(ll)]]))
    
    
        self.res_cost_figure=go.Figure(data=[self.result_cost_table]) 
        self.res_perf_measure_figure=go.Figure(data=[self.result_performance_measure_table])  
        self.res_parameter_figure=go.Figure(data=[self.result_parameter_table])     
    
    '''
    window_range: Array for batching windows (for hybrid and batching) - currently hybrid is no available
    tunning_range: Array for tunning parameters of threshold
    tunning_range_LP: Array for tunning parameters of LP (for values smaller than 0.9, running LP policy becomes super slow)
    alpha_range: Tunning parameter for hybrid(not available at the moment) --> does not affect the algorithms
    '''
    def generate_everything(self,train_times,test_times,train_dates,test_dates,window_range,tunning_range,tunning_range_LP,alpha_range,cbar_times,cbar_dates):
        ##Generate the results
        self.generate_results(train_times,test_times,train_dates,test_dates,window_range,tunning_range,tunning_range_LP,alpha_range,cbar_times,cbar_dates)    
        ##Show the results
        self.generate_table()
        self.res_cost_figure.show()
        self.res_perf_measure_figure.show()
        self.res_parameter_figure.show()
        self.res_cost_figure.write_image('result_table'+str(self.params[0]['cd'])+'test_day_'+test_dates[0][0:2]+'test_month_'+test_dates[0][3:5]+'test_time_'+test_times[0][0:2]+'.jpeg')
        self.res_parameter_figure.write_image('parameter_table'+str(self.params[0]['cd'])+'test_day_'+test_dates[0][0:2]+'test_month_'+test_dates[0][3:5]+'test_time_'+test_times[0][0:2]+'.jpeg')
        self.res_perf_measure_figure.write_image('performance_measure_table'+str(self.params[0]['cd'])+'test_day_'+test_dates[0][0:2]+'test_month_'+test_dates[0][3:5]+'test_time_'+test_times[0][0:2]+'.jpeg')

            
