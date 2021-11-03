# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 10:04:03 2021

@author: Sofia Gonzalez (sg6818)
"""

import oslo_sim as o
import matplotlib.pyplot as plt
import numpy as np
import math
import copy as copy
import scipy.optimize as opt
import logbin_kc as lb #taken from Max Falkenberg McGillivray - 2019 Complexity & Networks course
import pickle

def get_height_prob(heights):
    unique_heights, counts = np.unique(heights, return_counts=True)
    height_prob = counts/len(heights)
    return unique_heights, height_prob

power_law = lambda x, const, power: const * x ** power
linear_reg = lambda x, const, slope: slope * x + const

def chi_sq(theoretical, observed, unpacked= False):
    theoretical =np.array(theoretical)
    observed =np.array(observed)
    chi_sq = (np.abs(observed-theoretical)**2)/(theoretical)
    if unpacked: 
        return chi_sq, np.sum(chi_sq)/(len(chi_sq)-1)
    else:
        return np.sum(chi_sq)/(len(chi_sq)-1)

def get_avalanche_size_prob(aval_size):
    unique_sizes, counts = np.unique(aval_size, return_counts=True)
    aval_size_prob = counts/len(aval_size)
    return unique_sizes, aval_size_prob

def get_log_bins(data, bin_scaling=1.3, zeros=True):
    centres, freqs = lb.logbin(data, scale=bin_scaling, zeros=zeros) #taken from Max Falkenberg McGillivray - 2019 Complexity & Networks course
    return np.array(centres), np.array(freqs)

#%%
"""OSLO MODEL - RUN FOR DIFFERENT SIZES"""

p=0.5
iterations =13000

#create empty arrays that will be the arrays of arrays
all_height_o = []
all_avalanche_size = []
all_tc = []
all_last_th_means = []
all_slope_means =[]

small_power = math.trunc(int(input("Smallest system size (in power of 2):")))
large_power = math.trunc(int(input("Largest system size (in power of 2):")))

system_sizes = [2**i for i in range(small_power,large_power+1)]
print("SYSTEM SIZES:", system_sizes)
print("Minimum number of iterations for t_c:", system_sizes[-1]**2) #ensure tc is reached
iterations = math.trunc(int(input("Number of iterations desired:")))

for i in range(len(system_sizes)):
    
    print("SYSTEM SIZE BEING RUN:",system_sizes[i])
    
    height_o, avalanche_size, tc, slope_means, last_th_means = o.simulation(system_sizes[i], iterations, p)
    
    #append simulation arrays
    all_height_o.append(height_o)
    all_avalanche_size.append(avalanche_size)
    all_tc.append(tc)
    all_last_th_means.append(last_th_means)
    all_slope_means.append(slope_means)
    
oslo1run_save = open("NE6512/oslo1run.pkl" , "wb")
pickle.dump(all_height_o, oslo1run_save, pickle.HIGHEST_PROTOCOL)
pickle.dump(all_avalanche_size, oslo1run_save, pickle.HIGHEST_PROTOCOL)
pickle.dump(all_tc, oslo1run_save, pickle.HIGHEST_PROTOCOL)
pickle.dump(all_last_th_means, oslo1run_save, pickle.HIGHEST_PROTOCOL)
pickle.dump(all_slope_means, oslo1run_save, pickle.HIGHEST_PROTOCOL)
oslo1run_save.close()

#%%
"""TASK 1 - OSLO MODEL CHECKS"""


#%%
"""TASK 2.A) - PLOT PILE HEIGHT EVOLUTION"""

for i in range(len(system_sizes)):
    alpha=1-i*0.1
    plt.plot(all_height_o[i], alpha=alpha)
plt.xlabel("time (number of grains added)")
plt.ylabel("height of the pile")
plt.legend(system_sizes)
plt.show()

#%%
"""SMOOTHING DATA - SYSTEM AVERAGES OVER SEVERAL RUNS"""

num_runs=10

tilde_height_o = np.array(copy.deepcopy(all_height_o))
tilde_avalanche_size= np.array(copy.deepcopy(all_avalanche_size))
tilde_tc = np.array(copy.deepcopy(all_tc))
tilde_last_th_means = np.array(copy.deepcopy(all_last_th_means))

for j in range(1,num_runs):
    for i in range(len(system_sizes)):
        
        print("SYSTEM SIZE BEING RUN:", system_sizes[i])
        height_o, avalanche_size, tc, last_th_means = o.simulation(system_sizes[i], iterations, p)
        
        #work out means
        tilde_height_o[i]=(tilde_height_o[i]*j+height_o)/(j+1)
        tilde_avalanche_size[i]=(tilde_avalanche_size[i]*j+avalanche_size)/(j+1)
        tilde_tc[i]=(tilde_tc[i]*j+tc)/(j+1)
        tilde_last_th_means[i]=(tilde_last_th_means[i]*j+last_th_means)/(j+1)
        

"""PLOT NEW SMOOTHED HEIGHTS"""
for i in range(len(system_sizes)):
    alpha=1-i*0.1
    plt.plot(tilde_height_o[i], alpha=alpha)
plt.xlabel("time (number of grains added)")
plt.ylabel(r"$\tilde{h}$")
plt.legend(system_sizes)
plt.savefig(fname= "h_tilde.png",dpi=1000)
plt.show()

#%%
"""TASK 2. B) and C) -- AVERAGE HEIGHT VS SYSTEM SIZE"""

#plot average height vs system size
mean_tilde_heights =[]
std_tilde_heights=[]
for i in range(len(system_sizes)):
    mean_tilde_heights.append(np.mean(tilde_height_o[i][math.trunc(0.8*iterations):]))
    std_tilde_heights.append(np.std(tilde_height_o[i][math.trunc(0.8*iterations):]))
    
plt.xlabel("system size")
plt.ylabel(r"$\langle \tilde{height} post steady state")
plt.errorbar(system_sizes, mean_tilde_heights, yerr=std_tilde_heights, fmt='ro')
plt.plot(system_sizes, mean_tilde_heights)
plt.savefig(fname= "h_as_L.png",dpi=1000)
plt.show()

plt.xlabel("system size")
plt.ylabel(r"$\sigma_{\tilde{h}}$")
plt.plot(system_sizes, std_tilde_heights,'ro')
plt.plot(system_sizes, std_tilde_heights)
plt.savefig(fname= "h_std.png",dpi=1000)
plt.show()
#%%
"""TASK 2. B) and C) -- AVERAGE CROSS OVER TIME VS SYSTEM SIZE"""

plt.xlabel("system size")
plt.ylabel("average cross-over time")
plt.plot(system_sizes, tilde_tc,'ro')
plt.plot(system_sizes, tilde_tc)
plt.savefig(fname= "t_c_as_L.png",dpi=1000)
plt.show()

#%%
"""TASK 2. D) -- DATA COLLAPSE"""
#we plot h/L vs t/L^2
plt.xlabel(r"$t\,/\,L^2$")
plt.ylabel(r"$\tilde{h}\,/\, L$")
for i in range(len(system_sizes)):
    time_scaled = np.arange(0,iterations)/(system_sizes[i]**2)
    trunc=math.trunc(iterations*6/time_scaled[-1])
    plt.plot(time_scaled[:trunc], (tilde_height_o[i]/system_sizes[i])[:trunc])
plt.legend(system_sizes, title='system size (L)')
plt.savefig(fname= "data_collapse.png",dpi=1000)
plt.show()

final_val=[]
for i in range(len(system_sizes)):
    time_scaled = np.arange(0,iterations)/(system_sizes[i]**2)
    final_val.append(time_scaled[-1])

#%%
"""FIT CHECK FOR x<1 - just h"""
sys = -1 #system size we want to test it for. -1 is the last one (from indexing)
system = system_sizes[sys]
time= np.arange(0, all_tc[sys])
(const, power), covar = opt.curve_fit(power_law, time, all_height_o[sys][0:math.trunc(all_tc[sys])])

print("FITTED PARAMETERS:")
print("constant:", const)
print("power:", power)

plt.title("height evolution pre steady state- size:"+ str(system_sizes[sys]))
plt.xlabel(r"$t$ - pre steady state")
plt.ylabel(r"$h$")
plt.plot(time, all_height_o[sys][0:math.trunc(all_tc[sys])],color='red', label ='observed data')
plt.plot(time, power_law(time, const, power), color='black', label ='power law fit')
plt.legend()
plt.savefig(fname= "curvefit_presteady.png",dpi=1000)
plt.show()

#%%
"""FIT CHECK FOR x<1 - scaled, h/L and t/L^2"""
# sys = -1 #system size we want to test it for. -1 is the last one (from indexing)
# system = system_sizes[sys]
# time_scaled = np.arange(0,all_tc[sys])/(system_sizes[i]**2)
# (const, power), covar = opt.curve_fit(power_law, time_scaled, np.array(all_height_o[sys][0:math.trunc(all_tc[sys])])/system_sizes[i])

# print("FITTED PARAMETERS:")
# print("constant:", const)
# print("power:", power)

# plt.title("height evolution pre steady state- size:"+ str(system_sizes[sys]))
# plt.xlabel(r"$t/L^2$ - pre steady state")
# plt.ylabel(r"$h/L$")
# plt.plot(time_scaled,  np.array(all_height_o[sys][0:math.trunc(all_tc[sys])]/system_sizes[i]),color='red')
# plt.plot(time_scaled, power_law(time_scaled, const, power), label='fit', color='black')
# plt.legend(["observed data", "power law fit"])
# plt.savefig(fname= "curvefit_presteady_scaled.png",dpi=1000)
# plt.show()

#%%
"""TASK 2. E) --  <h>, STD(<h>), slopes and P(h;L)"""
#previously, in one of our checks, we had worked out <\tilde{h}> for the last 20% of the
#values, now we obtain it as required by the exercise
mean_heights=[]
std_heights=[]
mean_slopes=[]
std_slopes=[]
for i in range(len(system_sizes)):
    mean_heights.append(np.mean(all_height_o[i][math.trunc(all_tc[i]+1):]))
    std_heights.append(np.std(all_height_o[i][math.trunc(all_tc[i]+1):]))
    #note that average slope != average threshold slope
    mean_slopes.append(np.mean(all_slope_means[i][math.trunc(all_tc[i]+1):]))
    std_slopes.append(np.std(all_slope_means[i][math.trunc(all_tc[i]+1):]))

plt.xlabel("system size")
plt.ylabel("mean height post steady state")
plt.plot(system_sizes, mean_heights)
plt.plot(system_sizes, mean_heights, 'ro')
plt.savefig(fname= "h_as_L.png",dpi=1000)
plt.show()

plt.title("Stdev of <h> vs system size")
plt.xlabel("system size - log",  fontsize=16)
plt.ylabel(r"$\sigma_{h}$ - log", fontsize=16)
plt.loglog(system_sizes, std_heights)
plt.loglog(system_sizes,std_heights,'ro')
plt.savefig(fname= "loglog_std.png",dpi=1000)
plt.show()
#%%
"""TASK 2. E) - fitting for <h>"""
#ignoring the terms i>1 we use the following form for corrections to scaling
#<h>=a_o L (1-a1L^{-w1})

height_corrections = lambda L, a0, a1, omega1: a0 * L * (1 - a1 * L ** (-omega1))

(a0, a1, omega1), covar = opt.curve_fit(height_corrections, system_sizes, mean_heights, absolute_sigma=True)

print("FITTED PARAMETERS:")
print("a0:", a0)
print("a1:", a1)
print("omega 1:", omega1)

#from 'covar' (covariance matrix) we can calculate the standard deviation of the fit
print("stdev of fit:",np.sqrt(np.diag(covar))) #the off diagonal terms give the correlation of values

lengths = np.arange(1,math.trunc(1.1*system_sizes[-1]))
plt.title("Average height vs system size - data vs fit")
plt.xlabel("system size", fontsize=16)
plt.ylabel(r"$\langle h \rangle$ - post steady state", fontsize=16)
plt.plot(lengths, height_corrections(lengths, a0, a1, omega1), label='fit', color='black')
plt.plot(system_sizes, mean_heights,color='red', marker='x', linestyle='none', markersize=6)
plt.legend(["fit", "observed data"])
plt.savefig(fname= "mean_height_poststeady.png",dpi=1000)
plt.show()

# especific_sizes = height_corrections(np.array(system_sizes), a0, a1, omega1)
# plt.plot(system_sizes, abs(especific_sizes - np.array(mean_heights)), marker='x', linestyle='none', markersize=5)

"""SHOWING CORRECTIONS TO SCALING"""
plt.title("Average height vs system size - corrections to scaling")
plt.xlabel("system size (L)", fontsize=16)
plt.ylabel(r"$\langle h \rangle/{a_0 L}$", fontsize=16)
plt.plot(lengths, height_corrections(lengths, a0, a1, omega1)/(a0 * lengths), label='fit', color='black')
plt.plot(system_sizes, np.array(mean_heights)/(a0*np.array(system_sizes)),color='red', marker='x', linestyle='none', markersize=6)
plt.legend(["fit", "observed data"])
plt.savefig(fname= "mean_height_corr_scaling.png",dpi=1000)
plt.show()
#%%
"""TASK 2. F) - fitting for \sigma_h and \sigma_z"""

"""studying \sigma_h"""
(const, power), covar = opt.curve_fit(power_law, system_sizes, std_heights)

print("FITTED PARAMETERS:")
print("constant:", const)
print("power:", power)

lengths = np.arange(1,math.trunc(1.1*system_sizes[-1]))
plt.title(r"$\sigma_{<h>}$ vs system size - data vs fit")
plt.xlabel("system size", fontsize=16)
plt.ylabel(r"$\sigma_{<h>}$", fontsize=16)
plt.loglog(lengths, power_law(lengths, const, power), label='fit', color='black')
plt.loglog(system_sizes, std_heights,color='red',marker='x', linestyle='none', markersize=6)
plt.legend(["power law fit", "observed data"])
plt.savefig(fname= "std_fit.png",dpi=1000)
plt.show()

"""SHOWING CORRECTIONS TO SCALING"""
plt.title(r"$\sigma_{<h>}$ vs system size - correction to scaling")
plt.xlabel("system size (L)", fontsize=16)
plt.ylabel(r"$\sigma_{<h>}/{bL^{\gamma}}$", fontsize=16)
plt.plot(lengths, power_law(lengths, const, power)/(const*lengths**power), label='fit', color='black')
plt.plot(system_sizes, np.array(std_heights)/(const*np.array(system_sizes)**power),color='red',marker='x', linestyle='none', markersize=6)
plt.legend(["power law fit", "observed data"])
plt.axis((0, lengths[-1], 0.95, 1.05))
plt.savefig(fname= "std_corr_sc.png",dpi=1000)
plt.show()

"""studying \sigma_z"""

(const, power), covar = opt.curve_fit(power_law, system_sizes, std_slopes)

print("FITTED PARAMETERS:")
print("constant:", const)
print("power:", power)

lengths = np.arange(1,math.trunc(1.1*system_sizes[-1]))
plt.title(r"$\sigma_{z}$ vs system size - data vs fit")
plt.xlabel("system size", fontsize=16)
plt.ylabel(r"$\sigma_{z}$", fontsize=16)
plt.loglog(lengths, power_law(lengths, const, power), label='fit', color='black')
plt.loglog(system_sizes, std_slopes,color='red',marker='x', linestyle='none', markersize=6)
plt.legend(["power law fit", r"$\sigma_z$ data"])
plt.savefig(fname= "std_z_fit.png",dpi=1000)
plt.show()

#%%
"""TASK 2. G) - P(h;L)"""

gaussian = lambda x: 1/(np.sqrt(2*np.pi)) * np.exp(-0.5*(x**2))

unique_heights = []
height_probs = []
for i in range(len(system_sizes)):
    unique_height, height_prob = get_height_prob(all_height_o[i][math.trunc(all_tc[i]+1):])
    height_probs.append(height_prob)
    unique_heights.append(unique_height)

plt.title(r"Height probability distribution $P(h;L)$")
plt.ylabel(r"$P(h;L)$")
plt.xlabel(r"pile height ($h$)")
for i in range(len(system_sizes)):
    plt.plot(unique_heights[i], height_probs[i], linewidth=0.8)
plt.axis((0, math.trunc(1.1*all_height_o[-1][-1]), 0, 0.6))
plt.legend(system_sizes)
plt.show()

plt.title(r"Scaled $P(h;L)$")
plt.ylabel(r"$\sigma_h \, P(h;L)$")
plt.xlabel(r"$(h-\langle h \rangle)/{\sigma_h}$")
for i in range(len(system_sizes)):
    scaled_height = (np.array(unique_heights[i])-mean_heights[i])/(std_heights[i])
    scaled_height_prob = std_heights[i]*np.array(height_probs[i])
    plt.plot(scaled_height, scaled_height_prob , linewidth=0.5, label=system_sizes[i])
# plt.axis((0, math.trunc(1.1*all_height_o[-1][-1]), 0, 0.6))
x_axis=np.arange(-4,4,0.1)
plt.plot(x_axis, gaussian(x_axis), label='gaussian')
plt.legend()
plt.savefig(fname= "gaussians.png",dpi=1000)
plt.show()

#%%
"""TASK 2. G) - chi squared"""

chi_sq_val_arr=[]

for i in range(len(system_sizes)):
    scaled_height = (np.array(unique_heights[i])-mean_heights[i])/(std_heights[i])
    scaled_height_prob = std_heights[i]*np.array(height_probs[i])
    theoretical_gauss = gaussian(scaled_height)
    
    chi_sq_unpack, chi_sq_val = chi_sq(theoretical_gauss, scaled_height_prob, unpacked=True)
    chi_sq_val_arr.append(chi_sq_val)
    print("chi squared (per dof) for system size "+str(system_sizes[i]), chi_sq_val)
    plt.title(r" $\chi^2$ for L=" +str(system_sizes[i]))
    plt.plot(scaled_height, chi_sq_unpack)
    plt.show()

plt.plot(chi_sq_val_arr)

#%%
"""TASK 3. a) - test: plotting P_N (s;L) vs s"""

unique_sizes = []
av_size_probs = []
for i in range(len(system_sizes)):
    unique_size, av_size_prob = get_avalanche_size_prob(all_avalanche_size[i][math.trunc(all_tc[i]+1):])
    av_size_probs.append(av_size_prob)
    unique_sizes.append(unique_size)

for i in range(len(system_sizes)):
    sys = i
    plt.title(r"$P_N(s;L)$-N ="+str(len(all_avalanche_size[sys][math.trunc(all_tc[sys]+1):])))
    plt.ylabel(r"$P(s;L)$")
    plt.xlabel(r"avalanche size ($s$) for L="+str(system_sizes[sys]))
    plt.loglog(unique_sizes[sys], av_size_probs[sys], '.')
    centres, freqs = get_log_bins(all_avalanche_size[sys][math.trunc(all_tc[sys]+1):], bin_scaling=1.3)
    plt.loglog(centres, freqs)
    plt.show()

print("Approximate value of N=" +str(len(all_avalanche_size[sys][math.trunc(all_tc[sys]+1):])))
plt.title(r"$\tilde{P}_N(s;L)$")
plt.ylabel(r"$\tilde{P}(s;L)$")
plt.xlabel(r"avalanche size ($s$) for L="+str(system_sizes[sys]))
for i in range(len(system_sizes)):
    sys = i
    # plt.loglog(unique_sizes[sys], av_size_probs[sys], '.')
    centres, freqs = get_log_bins(all_avalanche_size[sys][math.trunc(all_tc[sys]+1):], bin_scaling=1.3)
    plt.loglog(centres, freqs)
plt.legend(system_sizes)
plt.savefig(fname= "logbinneddata.png",dpi=1000)
plt.show()

#%%
"""TASK 3.a) - curve fit for largest system size - no corrections to scaling"""

sys=-1
centres, freqs = get_log_bins(all_avalanche_size[sys][math.trunc(all_tc[sys]+1):], bin_scaling=1.25)
plt.loglog(centres, freqs, '.', markersize=3)
plt.show()

#by eye we choose which values to pick to then send to the curve fit
start=5
end=12
(const, ts), covar = opt.curve_fit(power_law, centres[start:-end], freqs[start:-end])

print("FITTED PARAMETERS:")
print("constant:", const)
print("power:", ts)

plt.loglog(centres[start:-end], power_law(centres[start:-end], const, ts))
plt.loglog(centres, freqs, '.', markersize=3)

#%%
"""TASK 3.a) - curve fit for largest system size - corrections to scaling"""

prob_corrections = lambda s, a0, tau_s,a1, omega1: a0 * (s **tau_s)* (1 + a1 * s ** (-omega1))

sys=-1
centres, freqs = get_log_bins(all_avalanche_size[sys][math.trunc(all_tc[sys]+1):], bin_scaling=1.25)
plt.loglog(centres, freqs, '.', markersize=3)
plt.show()

#by eye we choose which values to pick to then send to the curve fit
start=5
end=12
(a0, tau_s, a1, omega1), covar = opt.curve_fit(prob_corrections, centres[start:-end], freqs[start:-end])

print("FITTED PARAMETERS:")
print("constant:", a0)
print("power:", tau_s)

plt.loglog(centres[start:-end], prob_corrections(centres[start:-end], a0, tau_s, a1,omega1))
plt.loglog(centres, freqs, '.', markersize=3)


#%%
"""TASK 3.a) - data collapse"""

tau_s= 1.55
D=2.23

# print("Approximate value of N=" +str(len(all_avalanche_size[sys][math.trunc(all_tc[sys]+1):])))
plt.title("Scaling function - data collapse", fontsize=16)
plt.ylabel(r"$s^{\tau_s}\,\tilde{P}(s;L)$", fontsize=16)
plt.xlabel(r"$s/{L^D}$", fontsize=16)
for i in range(1,len(system_sizes)):
    sys = i
    centres, freqs = get_log_bins(all_avalanche_size[sys][math.trunc(all_tc[sys]+1):], bin_scaling=1.3,zeros=False)
    plt.loglog(centres/(system_sizes[sys]**D), (centres**tau_s)*freqs, label=system_sizes[sys])
plt.legend()
plt.savefig(fname= "datacollapseprobs.png",dpi=1000)
plt.show()

#%%
"""TASK 3.b) - kth moment plot"""

max_kth_moment = 4

all_kth_moments=[]
for i in range(len(system_sizes)):
    kth_moments=[]
    for k in range(1,(max_kth_moment+1)):
        kth_moments.append(np.mean(np.array(all_avalanche_size[i][math.trunc(all_tc[i]+1):], dtype='float64')**k))
    all_kth_moments.append(kth_moments)
    
all_kth_moments = np.array(all_kth_moments, dtype='float64')

plt.title(r"$k^{th}\; moment - \langle s^{k} \rangle$", fontsize=16)
plt.xlabel(r"system size $(L)$", fontsize=15)
plt.ylabel(r"$\langle s^{k} \rangle$", fontsize=15)
for k in range(max_kth_moment):
    label = "k="+str(k+1)
    plt.loglog(system_sizes, all_kth_moments[:,k], '.', label=label)
    #all_kth_moments[:,k] is the kth moment for all system sizes
plt.legend(fontsize=12)
plt.axis((0, math.trunc(1.5*system_sizes[-1]), 0, math.trunc(100*np.max(all_kth_moments))))
plt.savefig(fname= "kthmomments.png",dpi=600)
plt.show()

#%%
"""TASK 3.b) - kth moment curve fit"""
#we see that the kth moments follow a power law, we will try to find the value for
#the exponent

power_kth_moment = []
std_fit_power = []
start=0

logged_all_kth_moments= np.log(all_kth_moments)

for k in range(max_kth_moment):
    (const, power), covar= opt.curve_fit(linear_reg, np.log(system_sizes), logged_all_kth_moments[:,k])
    power_kth_moment.append(power)
    labelfit="fit k="+str(k+1)

    plt.loglog(system_sizes, power_law(system_sizes, np.exp(const), power), label=labelfit)
    label = "data k="+str(k+1)
    plt.loglog(system_sizes, all_kth_moments[:,k], '.', label=label)
plt.axis((0, math.trunc(1.5*system_sizes[-1]), 0, math.trunc(100*np.max(all_kth_moments))))
plt.legend()
plt.show()