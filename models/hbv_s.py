# import modules for simulation()
import numpy as np

#Simplyfied HBV (HBV-s) model
def simulation(data, params=[2.5, 250, -0.4, 1.6, 0.6, 0.055, 0.040, 1, 1]):
    """
    simulation fuction takes 2 arguments:
    1. data - pandas dataframe object with colums 'Temp' (in Celsius), 'Prec' (overall in mm/day), 'Evap' (in mm/day)
    2. params - list of 9 parameters of HBV-s model:
	    #beta=2.5  #soil
	    #fc=250    #soil
	    #Tt=-0.4   #snow formation temp
	    #gd=1.6    #day-degree value
	    #k=0.6     #evaporation correction parameter
	    #k0=0.055  #percolation to soilbox
	    #k1=0.040  #percolation through soil box
	    #Tk1=1     #threshold soil
	    #Tk2=1     #threshold groundwater
    """
    
    # daily temperature
    temp = data['Temp']
    # daily precipitation
    prec = data['Prec']
    # daily evaporation values
    evap  = data['Evap']
    
    #set parameters
    
    beta, fc, Tt, gd, k, k0, k1, Tk1, Tk2 = params
    
    #predefining containers
    #common
    n = len(temp) #number of timestamps
        #Snow
    snowbox = np.zeros(n)
    snowbox[0] = 0 #initial condition for snowbox
        #Soil
    soilbox = np.zeros(n)
    soilbox[0] = 140 #initial condition for soilbox
        #Free water in the system
    fwater = np.zeros(n)
    fwater[0] = 0 #initial condition for free water amount
        #Surface runoff box
    sw1 = np.zeros(n)
    sw1[0] = 0 #initial condition for surface water box
        #Groundwater runoff box
    gw1 = np.zeros(n)
    gw1[0] = 0 #initial condition for groundwater box
        #Recharge
    pq = np.zeros(n)
        #Effective evaporation
    ea = np.zeros(n) #effektiv evapo
        #Potential evaporation
    eaP = np.zeros(n) #potentiell evapo
        #Q
    q = np.zeros(n)
    
    for i in range(1, n):
        if temp[i]<Tt: #check snow formation or not
            snowbox[i]=snowbox[i-1]+prec[i]
            fwater[i]=0 #there are no melt water
        else: #temp>Tt
            snowbox[i] = np.array([snowbox[i-1]-gd*(temp[i]-Tt), 0]).max()
            fwater[i] = prec[i] + np.array([snowbox[i-1], gd*(temp[i]-Tt)]).min() - np.array([0, fwater[i-1]]).max()*0.1
    
        #evaporation control flow
        if (temp[i]>0) & (fwater[i]>0) == True:
            eaP[i] = k*evap[i-1]*temp[i]

            if eaP[i] <= fwater[i]:
                ea[i] = eaP[i]
            else:
                ea[i] = fwater[i]

        pq[i] = fwater[i]*((soilbox[i-1]/fc)**beta)  

        soilbox[i] = soilbox[i-1] + fwater[i] - pq[i] - ea[i]

        if sw1[i] < Tk1: #threshold effect conditions
            sw1[i] = sw1[i-1] + pq[i] - np.array([0, sw1[i-1]]).max()*0.08
        else:
            sw1[i] = sw1[i-1] + pq[i] - np.array([0, sw1[i-1]]).max()*k0

        if gw1[i] < Tk2:
            gw1[i] = gw1[i-1] + (sw1[i-1]*0.06) - gw1[i-1]*0.08
        else:
            gw1[i] = gw1[i-1] + (sw1[i-1]*k1) - gw1[i-1]*k1

        q[i] = np.array([0, sw1[i-1]]).max()*k0 + np.array([0, gw1[i-1]]).max()*k1

    return q

# import modules for interaction()
import pandas as pd
import sys
sys.path.append('../tools/')
from wfdei_to_lumped_dataframe import dataframe_construction
from metrics import NS

def interaction(river_name, path_to_scheme, path_to_observations, 
                beta, fc, Tt, gd, k, k0, k1, Tk1, Tk2):
    
    # simulate our modeled hydrograph
    data = dataframe_construction(path_to_scheme)
    data['Qsim'] = simulation(data, [beta, fc, Tt, gd, k, k0, k1, Tk1, Tk2])
    
    # read observations
    obs = pd.read_csv(path_to_observations, index_col=0, parse_dates=True, 
                      squeeze=True, header=None, names=['Date', 'Qobs'])
    
    # concatenate data 
    data = pd.concat([data, obs], axis=1)
       
    # calculate efficiency criterion
    # slice data only for observational period and drop NA values
    data_for_obs = data.ix[obs.index, ['Qsim', 'Qobs']].dropna()
    eff = NS(data_for_obs['Qobs'], data_for_obs['Qsim'])  
    
    # plot
    ax = data.ix[obs.index, ['Qsim', 'Qobs']].plot(figsize=(10, 7), style=['b-', 'k.'])
    ax.set_title(river_name + ' daily runoff modelling, ' + 'Nash-Sutcliffe efficiency: {}'.format(np.round(eff, 2)))
    #text_pos = np.max(data['Qobs'])
    #ax.text(obs.index[100], text_pos, 'NS: {}'.format(np.round(eff, 2)), size=14)
    
