# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:16:41 2021

@author: Sofia
"""
import numpy as np
# from logbin_kc import logbin

choices= np.arange(1,30)
array_to_bin = [np.random.choice(choices) for i in range(10)]
print(type(array_to_bin))
print("initial", array_to_bin)
unique, counts=np.unique(array_to_bin, return_counts= True)
print("unique", unique)
print("counts", counts)