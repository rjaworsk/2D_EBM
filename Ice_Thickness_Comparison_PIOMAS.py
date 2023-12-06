#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:37:24 2023

@author: ricij
"""

import matplotlib.pyplot as plt
import numpy as np
import Functions
from Tuning_Main import Mesh
import xarray as xr

filepath = "/Users/ricij/Documents/Universität/Master/Masterarbeit/PIOMAS.thick.daily.1979.2023.Current.v2.1.dat"

data = np.genfromtxt(filepath)

line, col = data.shape
tt = np.zeros((365,2))

u = 0
for t in range(1,366):
    for i in range(line):
   
        if data[i,1] ==t:
            tt[t-1,1] += data[i,2]
            tt[t-1,0] = t
            

              

# tt[:,1] = tt[:,1]/9

# zz = np.zeros((365,2))
# zz[:,0] = tt[:,0]
# zz[0:285,1] = tt[80:, 1]
# zz[285:,1] = tt[0:80,1]
# plt.plot(zz[:,0], zz[:,1], label="Ice Thickness (PIOMAS)")
# plt.plot(annual_mean_H_I_north,  label="Ice Thickness (Modell)")
# plt.xlim((0, 365 - 1))
# labels = ["March", "June", "September", "December", "March"]
# plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
# plt.grid()
# plt.legend(loc="upper right")
# plt.title("Ice Thickness Comparison")
# plt.tight_layout()
# plt.show()

mesh = Mesh()
filesea_north = "/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/Version_Paper_2D/netCDF_Data/seasurface_celsius_time_mean_year_grid_42_north_fldmean.nc"
filsea_south = "/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/Version_Paper_2D/netCDF_Data/seasurface_celsius_time_mean_year_grid_42_south_field_mean.nc"

nc_file = xr.open_dataset(filesea_north)
data_np_north = np.array(nc_file["seatemp"])

nc_file = xr.open_dataset(filsea_south)
data_np_south = np.array(nc_file["seatemp"])

plt.plot(data_np_north[:,0,0])
plt.plot(data_np_south[:,0,0])
print(np.mean(data_np_north[:,0,0]))
print(np.mean(data_np_south[:,0,0]))



filesea_north = "/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/Version_Paper_2D/netCDF_Data/seasurface_celsius_time_mean_year_grid_42_north.nc"
filsea_south = "/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/Version_Paper_2D/netCDF_Data/seasurface_celsius_time_mean_year_grid_42_south.nc"

 
nc_file = xr.open_dataset(filesea_north)
data_np_north = np.array(nc_file["seatemp"])
data_temp = np.zeros(data_np_north.shape)
lat = np.array(nc_file["lat"])
long = np.array(nc_file["lon"])

data_temp = np.flip(data_np_north,1)
avg_nrd = [Functions.calc_mean_north(data_temp[t,:,:], mesh.area) for t in range(12)]
avg_south = [Functions.calc_mean_south(data_temp[t,:,:], mesh.area) for t in range(12)]

print(avg_nrd,avg_south )
