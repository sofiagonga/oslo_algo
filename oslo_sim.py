# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 22:50:57 2021

@author: Sofia
"""

import numpy as np
import matplotlib.pyplot as plt
import math

"""FUNCTIONS"""
def thresholds_init(L,prob,posib=[1,2]):
    init_th=[np.random.choice(posib,p= prob) for i in range(L)]
    return init_th

def new_th(prob,posib=[1,2]):
    return np.random.choice(posib,p= prob)

def find_slopes(heights):
    return np.diff(-np.append(heights, 0))

def relax(site, L, heights, slopes):
    i = site
    
    if 0<i<L-1:
        heights[i]-=1
        heights[i+1]+=1
        slopes[i]-=2
        slopes[i+1]+=1
        slopes[i-1]+=1
        
    if i==0:
        heights[i]-=1
        heights[i+1]+=1
        slopes[i]-=2
        slopes[i+1]+=1
        
    if i==L-1:
        heights[i]-=1
        slopes[i]-=1
        slopes[i-1]+=1
    
    return slopes, heights

"""INITIALISE PARAMETERS"""

def simulation(L, iterations, p, scratch = True, slopes_started=0, heights_started=0, thresholds_started=0):
    
    """DEFINING STARTING ARRAY"""
    if scratch:
        thresholds=thresholds_init(L, prob=[p,1-p], posib=[1,2])
        slopes=np.zeros(L)
        heights=np.zeros(L)
    
    else:
        zeroes = np.zeros(len(slopes_started))
        heights = np.append(heights_started, zeroes)
        slopes = find_slopes(heights)
        thresholds= np.append(thresholds_started, thresholds_init(math.trunc(L/2), prob=[p,1-p]))
    
    size_arr=[]
    h_o=[]
    h_o_from_slopes=[]
    thresholds_means=[]
    slope_means=[]
    tc_checked = False
    
    """SIMULATION"""
    for j in range(iterations):
        #add a grain to site 0 (1 in theory)
        heights[0]+=1
        slopes[0]+=1
        size=0 #start the size of the avalache as 0
    
        while (slopes>thresholds).any(): #run the relaxation until no supercritical sites
            for i in range(L): #check one by one the sites
                if slopes[i]>thresholds[i]: #check if this is an unrelaxed site
                    slopes, heights = relax(i, L, heights, slopes)
                    if i==L-1 and not tc_checked: 
                        tc=j-1
                        tc_checked = True
                    size+=1
                    thresholds[i]=new_th(prob=[p,1-p])
    
        h_o.append(heights[0])
        h_o_from_slopes.append(np.sum(slopes))
        size_arr.append(size)
        thresholds_means.append(np.mean(thresholds))
        last_th_means = np.mean(thresholds_means[math.trunc(0.8*iterations):])
        slope_means.append(np.mean(slopes))
    
    if not tc_checked: 
        print("STEADY STATE FOR LENGTH", L, "NOT REACHED")
        tc = None
    return np.array(h_o), np.array(size_arr), tc, slope_means, last_th_means
