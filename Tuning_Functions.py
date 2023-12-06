#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:56:09 2023

@author: ricij
"""
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Test_Main_full import compute_equilibrium
import Functions
from scipy.optimize import curve_fit
from sklearn.linear_model import  LinearRegression
import copy
class P_atm:
    def __init__(self):        
        self.heat_capacity = 0.3 * np.ones((mesh.n_latitude, mesh.n_longitude)) #Atmosphere column heat capacity [W yr m^-2 degC^-1]
        self.diffusion_coeff = 2E14 * np.ones((mesh.n_latitude, mesh.n_longitude))  #Large-scale atmospheric diffusivity [m^2 yr^-1]
        
class P_ocn:
    def __init__(self):
        self.K_O = 4.4E11   #Large-scale ocean diffusivity [m^2 yr^-1]
        self.c_O = 1.27E-4  #Ocean specific heat capacity [W yr kg^-1 degC^-1]
        self.rhoo = 1025    #Density of sea water [kg m^-3]
        self.Hml_const = 75 #Mixed-layer depth when set constant [m]


# Here are all the functions which are being used to tune the model 

class Mesh_():
    def __init__(self):
        self.n_latitude  = 65
        self.n_longitude = 128
        self.ndof = self.n_latitude * self.n_longitude
        self.h = np.pi / (self.n_latitude - 1)
        self.area = Functions.calc_area(self.n_latitude, self.n_longitude)
        self.csc2 = np.array([1 / np.sin(self.h * j) ** 2 for j in range(1, self.n_latitude - 1)])
        self.cot = np.array([1 / np.tan(self.h * j) for j in range(1, self.n_latitude - 1)])
        
        self.RE = 6.37E6  #Radius of the earth
        self.Tf = -1.8    #Freezing point of sea water [degC]
        self.ki = 2.0     #Thermal conductivity of sea ice [W m^-1 degC^-1]
        self.Tm = -0.1    #Melting Temperature of sea water
        self.Lf = 10.0    #Latent heat of fusion of sea ice [W yr m^-3]
        
        self.A_up = 380   #Fluxes
        self.B_up = 7.9
        self.A_dn = 335
        self.B_dn = 5.9
        self.A_olr = 241
        self.B_olr = 2.4
        
        self.C_s = (1350 * 750 * 1  + 1.225 * 1000 * 3850) / 3.15576e7 # Heat Capacity Land
        

# As climate data come sin netcdf format we have to transform  it first 
def transform_net_cdf(file, variable='temp'): 
    nc_file = xr.open_dataset(file)
    data_np = np.array(nc_file[variable])
    data_temp = np.zeros(data_np.shape)
    lat = np.array(nc_file['lat'])
    long = np.array(nc_file['lon'])
    # Have to flip the array because longitude starts at 0
    data_temp[:,:,:64] = data_np[:,:,64:]
    data_temp[:,:,64:] = data_np[:,:,:64]
    data_temp = np.flip(data_temp,1)
    return data_temp, lat, long
    
# def transform_net_cdf(file, variable='temp'): 
#     nc_file = xr.open_dataset(file)
#     data_np = np.array(nc_file[variable])
#     return data_np


    
def Geo_dat_from_NCDF(data, mesh):
    Geo_dat = np.zeros((mesh.n_latitude, mesh.n_longitude))
    for i in range(mesh.n_latitude):
        for j in range(mesh.n_longitude): 
              if np.isnan(data[0,i,j]):
                  Geo_dat[i,j] = 5
              else: 
                  Geo_dat[i,j] = 1
                                    
    return Geo_dat 
    
   
def surface_temp_land(T_S_land, X1,X2,X3,X4): #calc surface temp of land without diffusion
    T_S_land = np.zeros((65, 128))
    T_ATM = np.zeros((65, 128))
    delta_t = 1/48
    true_longitude = Functions.calc_lambda(dt,  48,  ecc =  0.016740, per = 1.783037)
    insolation = Functions.calc_insolation(phi, true_longitude)
    phi_index_s = np.zeros((mesh.n_longitude))
    phi_index_n = np.zeros((mesh.n_longitude))
    coalbedo = Functions.calc_coalbedo(Geo_dat, phi_index_s, phi_index_n) 
    solar_forcing  = Functions.calc_solar_forcing(insolation[:,0],coalbedo, mesh)    
   
    #RHS = 1/heat_capacity * (solar_forcing - 2.15 *T_S_land - 210.3) # From lecture
   # RHS = 1/mesh.C_s * (solar_forcing + mesh.B_dn * T_ATM - mesh.A_up  + mesh.A_dn - mesh.B_up * T_S_land )
    RHS = 1/mesh.C_s * (solar_forcing + X1 * T_ATM - X4  + X3 - X2* T_S_land )
    
    T_S_land_new = T_S_land + delta_t * RHS
    
    #T_S_land_avg_ = [Functions.calc_mean_ocn(T_S_land_new[t, :], mesh.area) for t in range(48)]
          
    return T_S_land_new.ravel() 
   

    
def test_compute_equ(T_S_land,X1,X2,X3,X4):
    mesh = Mesh_()
    test_file = "/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/Version_Paper_2D/netCDF_Data/Era5_land_t2m_celsius_grid.nc"    
    data, lat , long = transform_net_cdf(test_file)
    Geo_dat = Geo_dat_from_NCDF(data, mesh)
    
    Ocean_boundary = Functions.Get_Ocean_Boundary_Distribution(Geo_dat)
    Surface_boundary = Functions.Get_Surface_Boundary_Distribution(Geo_dat)

    mesh = Mesh_()
     
    heat_capacity_s = Functions.calc_heat_capacity(Geo_dat)
    diffusion_coeff_s = np.ones((mesh.n_latitude, mesh.n_longitude)) * 0.18
    
    Lakes =  Functions.LandDstr_wLakes(Geo_dat)
    
    ntimesteps = 48
    dt = 1/ ntimesteps
    true_longitude = Functions.calc_lambda(dt,  ntimesteps, ecc =  0.016740, per = 1.783037)
    
   
    co2_ppm = 315.0
    c02_warming = Functions.calc_radiative_cooling_co2(co2_ppm)
  
    
    phi = np.linspace(np.pi/2,-np.pi/2,mesh.n_latitude) #from 90° south to 90° north
    
    #P_atm = P_atm() #Parameters for the atmosphere
    #P_ocn = P_ocn() #Parameters for the ocean
    
    diffusion_coeff_atm = P_atm.heat_capacity * P_atm.diffusion_coeff /mesh.RE**2 #Diffusion coefficient
    
    heat_capacity_ocn = P_ocn.c_O * P_ocn.rhoo * P_ocn.Hml_const * np.ones((mesh.n_latitude, mesh.n_longitude))  # Hml can also be variable
    diffusion_coeff_ocn = heat_capacity_ocn * P_ocn.K_O / mesh.RE**2  #Diffusion coefficient
    T_ATM, T_S_land, T_S, T_OCN, H_I, Surface_Temp, phi_index_s, phi_index_n  = compute_equilibrium(mesh, diffusion_coeff_atm, heat_capacity_atm, P_ocn, diffusion_coeff_ocn, heat_capacity_ocn,  phi, true_longitude,n_timesteps, Geo_dat, heat_capacity_s, diffusion_coeff_s, Ocean_boundary, Lakes, Surface_boundary,c02_warming, X1,X2,X3,X4, T_S_zero, data)

    T_land_nan = copy.copy(T_S_land)
    T_land_nan[T_land_nan==0] = np.nan 
    T_S_land_avg = [Functions.calc_mean(T_land_nan[:, :, t], mesh.area) for t in range(ntimesteps)]
    
    return T_S_land_avg 
 
 
if __name__ == '__main__':
    
  mesh = Mesh_() 
  test_file = "/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/Version_Paper_2D/netCDF_Data/Era5_land_t2m_celsius_grid.nc"    
  data , lat, long= transform_net_cdf(test_file)
  t2m = np.mean(data, axis = 0)
  Geo_dat = Geo_dat_from_NCDF(data, mesh)
  P_atm = P_atm()
  P_ocn = P_ocn()
  ntimesteps = 48
  dt = 1/ ntimesteps
  phi = np.linspace(np.pi/2,-np.pi/2,mesh.n_latitude)
  true_longitude = Functions.calc_lambda(dt,  ntimesteps, ecc =  0.016740, per = 1.783037)
  insolation = Functions.calc_insolation(phi, true_longitude)
  phi_index_s = np.zeros((mesh.n_longitude))
  phi_index_n = np.zeros((mesh.n_longitude))
  coalbedo = Functions.calc_coalbedo(Geo_dat, phi_index_s, phi_index_n) #new coalbdeo dependant on the ice_edge 
  
  solar_forcing  = Functions.calc_solar_forcing(insolation[:,0],coalbedo, mesh)        
 
  
  X1 = mesh.B_dn
  X2 = mesh.B_up 
  X3 = mesh.A_dn 
  X4 = mesh.A_up
  
  test_file = "/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/Version_Paper_2D/netCDF_Data/Era5_land_t2m_celsius_grid.nc"    
  data, lat , long = transform_net_cdf(test_file)
  plt.imshow(data[0,:,:])
  plt.show()
  T_S_land_avg_ = [Functions.calc_mean_ocn(data[t, :], mesh.area) for t in range(48)] #da Daten im Januar starten
  
  Geo_dat = Geo_dat_from_NCDF(data, mesh) # Only differentiate between land and ocean
  x, y= np.mgrid[0:65,0:128]
     
  xdata = np.vstack((x.ravel(), y.ravel()))
  T_ATM = np.zeros((65,128))
  T_S_land = np.zeros((65,128))
  T_S_zero = copy.copy(data[0,:,:])
  T_S_zero[np.isnan(T_S_zero)] = 0
  
  
  popt, pcov = curve_fit(surface_temp_land, xdata, T_S_zero.ravel())
  
  # file = "/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/Version_Paper_2D/netCDF_Data/snow_depth_timeMean_grid.nc"
  # snow_cover, lat1, long1 = transform_net_cdf(file, variable='schnee_hoehe')
  # snow_cover = np.mean(snow_cover, axis=0)
  # Snow_cover_zero = snow_cover
  # Snow_cover_zero[np.isnan(Snow_cover_zero)] = 0
  
  # t2m_zero = t2m
  # t2m_zero[np.isnan(t2m_zero)] = 0
  
  # model = LinearRegression()
  # model.fit(t2m_zero, snow_cover)
  
  # intercept = np.mean(model.intercept_)
  # slope = np.mean(model.coef_)
  # print(intercept )
  # print(slope)
  # inte = model.intercept_
  # pred = model.predict(t2m_zero)
