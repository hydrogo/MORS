#Simplyfied HBV (HBV-s) model
import numpy as np

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
    fwater[0] = 0 #initial condirion for free water amount
        #Surface runoff box
    sw1 = np.zeros(n)
    sw1[0] = 0 #initial condition for surface water box
        #Groundwater runoff box
    gw1 = np.zeros(n)
    gw1[0] = 0 #initial condition for groundwater box
        #PQ?
    pq = np.zeros(n)
        #Effective evaporation
    ea = np.zeros(n) #effektiv evapo
        #Potential evaporation
    eaP = np.zeros(n) #potentiell evapo
        #Q
    q = np.zeros(n)
    
    for i in range(1, n):
        if temp[i]<Tt: #will snow form?
            snowbox[i]=snowbox[i-1]+prec[i]
            fwater[i]=0 #no melt water
        else: #temp>Tt
            snowbox[i] = np.array([snowbox[i-1]-gd*(temp[i]-Tt), 0]).max()
            fwater[i] = prec[i] + np.array([snowbox[i-1], gd*(temp[i]-Tt)]).min() - np.array([0, fwater[i-1]]).max()*0.1
    
        #evapo control
        if (temp[i]>0) & (fwater[i]>0) == True:
            eaP[i] = k*evap[i-1]*temp[i]

            if eaP[i] <= fwater[i]:
                ea[i] = eaP[i]
            else:
                ea[i] = fwater[i]

        pq[i] = fwater[i]*((soilbox[i-1]/fc)**beta)  

        soilbox[i] = soilbox[i-1] + fwater[i] - pq[i] - ea[i]

        if sw1[i] < Tk1: #threshold effect
            sw1[i] = sw1[i-1] + pq[i] - np.array([0, sw1[i-1]]).max()*0.08
        else:
            sw1[i] = sw1[i-1] + pq[i] - np.array([0, sw1[i-1]]).max()*k0

        if gw1[i] < Tk2:
            gw1[i] = gw1[i-1] + (sw1[i-1]*0.06) - gw1[i-1]*0.08
        else:
            gw1[i] = gw1[i-1] + (sw1[i-1]*k1) - gw1[i-1]*k1

        q[i] = np.array([0, sw1[i-1]]).max()*k0 + np.array([0, gw1[i-1]]).max()*k1

    return q

