#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 09:14:58 2023

@author: ricij
"""
import copy
import numpy as np
import time
from scipy import sparse
import scipy.sparse.linalg as lg
from scipy import special
from scipy.optimize import curve_fit
import os

import matplotlib.pyplot as plt
import Functions
import ATM_test
import OCN_fct
import Surface_fct
from scipy.optimize import least_squares


class Mesh():
    def __init__(self):
        self.n_latitude  = 65 
        self.n_longitude =  128
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
        self.C_snow = 0.155026
        

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

class TimerError(Exception):

    """A custom exception used to report errors in use of Timer class"""

class Timer:
    
    
    
    """ Timer class from https://realpython.com/python-timer/#python-timer-functions """
    
    def __init__(self):

        self._start_time = None


    def start(self):

        """Start a new timer"""

        if self._start_time is not None:

            raise TimerError(f"Timer is running. Use .stop() to stop it")


        self._start_time = time.perf_counter()


    def stop(self, text=""):

        """Stop the timer, and report the elapsed time"""

        if self._start_time is None:

            raise TimerError(f"Timer is not running. Use .start() to start it")


        elapsed_time = time.perf_counter() - self._start_time

        self._start_time = None

        print("Elapsed time ("+text+f"): {elapsed_time:0.10e} seconds")
  
def ice_edge(H_I, phi, Geo_dat): # Calculating Ice Edge 
    nlatitude, nlongitude = H_I.shape
    ice_edge_latitude_n = np.zeros(nlongitude)
    ice_edge_latitude_s = np.zeros(nlongitude)
    lat_index_n = np.zeros(nlongitude)
    lat_index_s = np.zeros(nlongitude)
    
    # Calculation follows the paper of Aylmer
    for i in range(nlongitude): #calculation of ice edge latitude for each longitude
        
        if H_I[0,i] == 0 and (Geo_dat[0,i] == 5 or Geo_dat[0,i] == 2 ): #calc ice edge north 
            index_n = 0
            ice_latitude_n = phi[0]
            
        elif H_I[0,i] != 0 or (Geo_dat[0,i] == 3 or Geo_dat[0,i] == 1) :
            index_n = 1
            
            while (H_I[index_n,i] > 0 or (Geo_dat[index_n,i] == 3 or Geo_dat[index_n,i] == 1) and index_n<64): 
              index_n = index_n + 1

            for ind_temp in range(index_n-1,-1, -1):    #need extra loop in case there is land inbetween ice 
               if H_I[ind_temp,i] == 0 or (Geo_dat[ind_temp,i] == 3 or Geo_dat[ind_temp,i] == 1):
                ind_temp -=1 
               else: 
                ice_latitude_n = phi[ind_temp]
                index_n = ind_temp 
                break
               if ind_temp == 0: 
                ice_latitude_n = phi[ind_temp]
                index_n = ind_temp  
               
                
        if H_I[nlatitude-1,i] == 0  and (Geo_dat[nlatitude-1,i] == 5 or Geo_dat[nlatitude-1,i] == 2): #calc ice edge south
            index_s =nlatitude-1
            ice_latitude_s = phi[nlatitude-1]
            
        elif H_I[nlatitude-1,i] != 0 or (Geo_dat[nlatitude-1,i] == 3 or Geo_dat[nlatitude-1,i] == 1): 
            index_s = nlatitude-2
            
            while (H_I[index_s,i] > 0 or (Geo_dat[index_s,i] == 3 or Geo_dat[index_s,i] == 1) and index_s >0): 
              index_s = index_s - 1
            
            for ind_temp in range(index_s+1,nlatitude):   #need extra loop in case there is land inbetween ice 
               if H_I[ind_temp,i] == 0  or (Geo_dat[ind_temp,i] == 3 or Geo_dat[ind_temp,i] == 1):
                ind_temp +=1 
               else: 
                ice_latitude_s = phi[ind_temp]
                index_s = ind_temp 
                break
               if ind_temp == nlatitude-1: 
                ice_latitude_s = phi[nlatitude-1]
                index_s = nlatitude-1   
                
        ice_edge_latitude_n[i] = ice_latitude_n   
        ice_edge_latitude_s[i] = ice_latitude_s   
        lat_index_n[i] = index_n
        lat_index_s[i] = index_s
    return lat_index_s, ice_edge_latitude_s, lat_index_n, ice_edge_latitude_n
  
def time_conv(t):    
    if 37<=t<=40:
        mon = 0
    elif 41<=t<=44:
        mon = 1
    elif 45<=t<=47 or t ==0:
        mon = 2
    elif 1<=t<=4: 
        mon = 3
    elif 5<=t<=8: 
        mon = 4
    elif 9<=t<=12: 
        mon =5
    elif 13<=t<=16: 
        mon = 6
    elif 17<=t<=20: 
        mon = 7
    elif 21<=t<=24: 
        mon = 8
    elif 25<=t<=28: 
        mon = 9
    elif 29<=t<=32: 
        mon = 10
    elif 33<=t<=36:
        mon = 11
    return mon     
    
def snow_edge(Geo_dat, mesh, phi): 
    snow_edge_north = np.zeros(mesh.n_longitude)
    for j in range(mesh.n_longitude):
        for i in range(int(mesh.n_latitude/2)):
            if Geo_dat[i,j] == 3: 
                snow_edge_north[j]= phi[i]
        if  snow_edge_north[j] == 0:     
            snow_edge_north[j]= phi[0]
                
    return snow_edge_north         

def surface_temp_land(T_ATM, solar_forcing, phi, mesh, Geo_dat, T_S_land, delta_t, heat_capacity, Aup, Bup, Adn, Bdn): #calc surface temp of land without diffusion
   # T_S_land_new = np.zeros((mesh.n_latitude, mesh.n_longitude))
    RHS = np.zeros((mesh.n_latitude, mesh.n_longitude))
  
    for i in range(mesh.n_latitude):
        for j in range(mesh.n_longitude):
            if Geo_dat[i,j] == 3:
                   
                    RHS[i,j] = 1/heat_capacity_s[1] * (solar_forcing[i,j] + (Bdn) * T_ATM[i,j] - Aup  + Adn - Bup * T_S_land[i,j]) 
                    
                    if i > 32: 
                        
                      RHS[i,j] = 1/heat_capacity_s[1] * (solar_forcing[i,j] -200 + (Bdn) * T_ATM[i,j] - Aup  + Adn - (Bup) * T_S_land[i,j])
                        

                    
            elif Geo_dat[i,j] == 1:  
                    RHS[i,j] = 1/heat_capacity_s[0] * (solar_forcing[i,j] + (Bdn) * T_ATM[i,j] - Aup  + Adn - Bup * T_S_land[i,j]) 
                    
            else: 
                RHS[i,j] = 0
          
    T_S_land_new = T_S_land + delta_t * RHS
        
    return T_S_land_new

def surface_temp_ocean(T_ATM, T_OCN, H_I, solar_forcing, phi, mesh, Geo_dat): #calculate surface temp of ocean
    T_S = copy.copy(T_OCN)
    # Calculation follows the paper of Aylmer
    if H_I.any() > 0 : #sea ice exists
    
        phi_index_s, phi_i_s, phi_index_n, phi_i_n  = ice_edge(H_I,phi, Geo_dat)
        
        T_d = (mesh.ki * mesh.Tf + H_I * (solar_forcing - mesh.A_up + mesh.A_dn + mesh.B_dn * T_ATM))/(mesh.ki + mesh.B_up * H_I)
        
        for i in range(mesh.n_longitude):
            
            for j in range(mesh.n_latitude):
                
                if H_I[j,i] > 0: 
                    
                   T_S[j,i] = mesh.Tm * (T_d[j,i] > mesh.Tm) + T_d[j,i] * (T_d[j,i] <= mesh.Tm)
                      
    return T_S 

def FreezeAndMelt(T_OCN, H_I, Hml, mesh): #calc new ice distribution
    T_OCN_new = copy.copy(T_OCN)
    H_I_new = copy.copy(H_I)
    z = mesh.Lf/(P_ocn.c_O*P_ocn.rhoo*Hml)
   
    # Calculation follows the paper of Aylmer
    for i in range(mesh.n_longitude):
      for j in range(mesh.n_latitude):   
         if H_I[j,i] < 0:
           
           H_I_new[j,i] = 0
           T_OCN_new[j,i] = T_OCN[j,i] - z[j,i]*H_I[j,i]
           
           if T_OCN_new[j,i] < mesh.Tf:
               H_I_new[j,i] = (mesh.Tf-T_OCN_new[j,i])/z[j,i]
               T_OCN_new[j,i] = mesh.Tf
        
         elif( H_I[j,i] == 0 and T_OCN[j,i] < mesh.Tf):
           
               H_I_new[j,i] = (mesh.Tf-T_OCN[j,i])/z[j,i]
               T_OCN_new[j,i] = mesh.Tf
       
           
         elif H_I[j,i] > 0:
           H_I_new[j,i] = H_I[j,i] + (mesh.Tf-T_OCN[j,i])/z[j,i]
           T_OCN_new[j,i] = mesh.Tf
      
           if H_I_new[j,i] < 0:
               T_OCN_new[j,i] = mesh.Tf -z[j,i]*H_I_new[j,i]
               H_I_new[j,i] = 0
     
    return T_OCN_new, H_I_new


def timestep_euler_forward_H_I(mesh,T_S, T_ATM, Fb, solar_forcing, H_I, t, delta_t): #calc new ice thickness 
    # Calculation follows the paper of Aylmer
    H_I_new = H_I - delta_t * (1/mesh.Lf * (-mesh.A_up - mesh.B_up * T_S + mesh.A_dn + mesh.B_dn * T_ATM + Fb + solar_forcing) * (H_I >0))
    
    return H_I_new


def Snow_melt(Geo_dat, T_ATM, T_S_land , mesh, phi_index_n, phi_index_s ): # Calculate if there is snow present from a given surface temperature
#     # def Model(t2m): 
#     #     return 1.54425e-14 - 0.00577 * t2m

#     # pred = Model(T_S_land)
#     max_eis = np.max(phi_index_n)
#     for i in range(mesh.n_latitude):  
#       for j in range(mesh.n_longitude):
#               if j-5<0 or j-1 <0: 
#                   jp1 = 0
#               else: 
#                   jp1 = j-5
#               if  j-1 <0: 
#                   jp2 = 0
#               else: 
#                   jp2 = j-1   
#               if  j+1 >127: 
#                   jp3 =127
#               else: 
#                   jp3 = j+1   
#               if  j+5 >127: 
#                     jp4 =127
#               else: 
#                     jp4 = j+5    
              
#               if ( i <=  np.max(phi_index_n[jp1:jp4]) ) and  Geo_dat[i,j] == 1 and T_S_land[i,j] < 0: 
#               #if ( i <=  max_eis) and  Geo_dat[i,j] == 1 and T_S_land[i,j] < 0:    
#                   Geo_dat[i,j] = 3
#               elif  ( 32 >i >  np.max(phi_index_n[jp1:jp4]) and phi_index_n[j] > 2 ) and Geo_dat[i,j] == 3 and T_S_land[i,j] > 0: 
#               #elif ( ( 32 >i >  max_eis) and phi_index_n[j] > 2 ) and Geo_dat[i,j] == 3 and T_S_land[i,j] > 0: 
                
#                   Geo_dat[i,j] = 1
    
    # for i in range(int(mesh.n_latitude/2)):
    #     for j in range(mesh.n_longitude):
    #         if T_S_land[i,j] < -1: 
    #             Geo_dat[i,j] = 3
    #         elif  T_S_land[i,j] > 0: 
    #             Geo_dat[i,j] = 1
                  
     return Geo_dat              
              
def compute_equilibrium(mesh, diffusion_coeff_atm, heat_capacity_atm, P_ocn, diffusion_coeff_ocn, heat_capacity_ocn,  phi, true_longitude,n_timesteps, Geo_dat0, heat_capacity_s,  Ocean_boundary, Lakes, Surface_boundary,  c02_warming, Aup, Bup, Adn, Bdn,Aolr, Bolr, Albdeo_ERA5, Albedo_up, Albedo_dn, max_iterations=50, rel_error=1e-5, verbose=True):
    # Step size
    delta_t = 1 / ntimesteps
    
    #Inital Conditions
    T_ATM = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    T_OCN = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    T_S_Ocean = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    T_S_land = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    H_I = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    Surface_Temp = np.zeros((mesh.n_latitude,mesh.n_longitude, ntimesteps))
    solar_forcing = np.zeros((mesh.n_latitude, mesh.n_longitude,ntimesteps))
    
    # Area-mean in every time step
    temp_atm = np.zeros(ntimesteps)
    temp_ocn=  np.zeros(ntimesteps)
    temp_s =  np.zeros(ntimesteps)
    temp_H_I =  np.zeros(ntimesteps)
    temp_land = np.zeros(ntimesteps)
    
    phi_index_s = np.zeros((mesh.n_longitude,ntimesteps))
    phi_index_n = np.zeros((mesh.n_longitude,ntimesteps))
    Geo_dat = np.zeros((mesh.n_latitude,mesh.n_longitude, ntimesteps))
    Geo_dat[:,:,-1] = Geo_dat0
    Snow_edge = np.zeros((mesh.n_longitude,ntimesteps))
   

    # Average temperature over all time steps from the previous iteration to approximate the error
    old_avg_atm = 0
    old_avg_ocn = 0
    old_avg_s = 0
    old_avg_H_I = 0
    old_avg_land = 0
    
    Fb = OCN_fct.BasalFlux(phi, mesh) #additional ocean flux
    Hml = P_ocn.Hml_const * np.ones((mesh.n_latitude,mesh.n_longitude)) # Can be changed 
    
    # Construct and factorize Jacobian for the atmosphere
    jacobian_atm = ATM_test.calc_jacobian_atm(mesh, diffusion_coeff_atm, P_atm.heat_capacity, phi, Geo_dat0, Aup, Bup, Adn, Bdn, Aolr, Bolr)
    m, n = jacobian_atm.shape
    eye = sparse.eye(m, n, format="csc")
    jacobian_atm = sparse.csc_matrix(jacobian_atm)
    jacobian_atm =lg.factorized(eye - delta_t * jacobian_atm)   
    
    # Construct and factorize Jacobian for the ocean
    jacobian_ocn = OCN_fct.calc_jacobian_ocn(mesh, diffusion_coeff_ocn, heat_capacity_ocn, phi, Ocean_boundary)
    m, n = jacobian_ocn.shape
    eye = sparse.eye(m, n, format="csc")
    jacobian_ocn = sparse.csc_matrix(jacobian_ocn)
    jacobian_ocn = lg.factorized(eye - delta_t * jacobian_ocn)
    
    # Compute insolation
    insolation = Functions.calc_insolation(phi, true_longitude)
    
    
    timer = Timer()
    for i in range(max_iterations):
        print("Iteration: ",i)
        timer.start()
        for t in range(ntimesteps):    
            
            phi_index_s[:,t], phi_i_s, phi_index_n[:,t], phi_i_n = ice_edge(H_I[:,:,t-1], phi, Geo_dat[:,:,t-1])  # new Ice_Edge Index 
            
            Geo_dat[:,:,t] = Functions.change_geo_dat(Geo_dat[:,:,t-1], H_I[:,:,t-1], mesh) #calc new ice distribution
             
            Snow_edge[:,t]= snow_edge(Geo_dat[:,:,t], mesh, phi)
            
            coalbedo = Functions.calc_coalbedo(Geo_dat[:,:,t], phi_i_n, phi_i_s, Snow_edge[:,t], Albedo_up, Albedo_dn ) #new coalbdeo dependant on the ice_edge #Albdeo_ERA5[mon,:,:]
            
            solar_forcing[:,:,t-1]  = Functions.calc_solar_forcing(insolation[:,t-1],coalbedo, mesh)        
            
            T_ATM[:,:,t] = ATM_test.timestep_euler_backward_atm(jacobian_atm, 1/ntimesteps, T_ATM[:,:,t-1], Surface_Temp[:,:,t-1], mesh, P_atm.heat_capacity, c02_warming, Geo_dat[:,:,t], Aup, Bup, Adn, Bdn, Aolr, Bolr)
            
            T_OCN[:,:,t] = OCN_fct.timestep_euler_backward_ocn(jacobian_ocn, 1/ntimesteps, T_OCN[:,:,t-1], T_S_Ocean[:,:,t-1], T_ATM[:,:,t-1], mesh, heat_capacity_ocn, solar_forcing[:,:,t-1], Fb, H_I[:,:,t-1], Geo_dat[:,:,t], Lakes)
            
            H_I[:,:,t] = timestep_euler_forward_H_I(mesh,T_S_Ocean[:,:,t-1], T_ATM[:,:,t-1], Fb, solar_forcing[:,:,t-1], H_I[:,:,t-1], t, delta_t)
             
            T_OCN[:,:,t], H_I[:,:,t] = FreezeAndMelt(T_OCN[:,:,t], H_I[:,:,t], Hml, mesh)
            
            solar_forcing_new = Functions.calc_solar_forcing(insolation[:,t],coalbedo, mesh)   
            
            T_S_land[:,:,t] = surface_temp_land(T_ATM[:,:,t], solar_forcing_new, phi, mesh, Geo_dat[:,:,t], T_S_land[:,:,t-1], delta_t, heat_capacity_s, Aup, Bup, Adn, Bdn) #without diffusion
           
            T_S_Ocean[:,:,t] = surface_temp_ocean(T_ATM[:,:,t], T_OCN[:,:,t], H_I[:,:,t], solar_forcing_new, phi, mesh, Geo_dat[:,:,t]) #original surface temp from Ocean Model --> is just for the ocean surface temperatue
           
            Surface_Temp[:,:,t] = Functions.unite_surface_temp(T_S_Ocean[:,:,t] , T_S_land[:,:,t] , mesh, Geo_dat[:,:,t])
            
            temp_atm[t] = Functions.calc_mean(T_ATM[:,:,t], mesh.area)
            temp_ocn[t] = Functions.calc_mean_ocn(T_OCN[:,:,t], mesh.area)
            temp_s[t] = Functions.calc_mean(Surface_Temp[:,:,t], mesh.area)
            temp_H_I[t] = Functions.calc_mean_ocn(H_I[:,:,t], mesh.area)
            temp_land[t] = Functions.calc_mean_ocn(T_S_land[:,:,t], mesh.area)
            
            
        timer.stop("one year")
        avg_temperature_atm = np.sum(temp_atm) / ntimesteps
        avg_temperature_ocn = np.sum(temp_ocn) / ntimesteps
        avg_temperature_s = np.sum(temp_s) / ntimesteps
        avg_H_I = np.sum(temp_H_I) / ntimesteps
        avg_temperature_land = np.sum(temp_land) / ntimesteps
      
        print("Fehler Eisdicke: ",np.abs(avg_H_I - old_avg_H_I))
        print("Fehler ATMO: ",np.abs(avg_temperature_atm - old_avg_atm))
        print("Fehler Surface: ",np.abs(avg_temperature_s - old_avg_s))
        print("Fehler Ocean: ",np.abs(avg_temperature_ocn - old_avg_ocn))
        print("Fehler Land: ",np.abs(avg_temperature_land - old_avg_land))
        print("Aup: ",Aup, " Bup: ", Bup, " Adn: ", Adn, " Bdn: ",Bdn)
        #print("A_OLR: ",Aolr, "B_OLR: ", Bolr )
       # print("Albedo_up: ", Albedo_up, "Albedo_dn", Albedo_dn)
        #print("Heat_cap: ", heat_capacity_s)
        if (np.abs(avg_temperature_atm - old_avg_atm)< rel_error) and (np.abs(avg_temperature_ocn - old_avg_ocn)< rel_error)  and  (np.abs(avg_temperature_s - old_avg_s)< rel_error) and (np.abs(avg_H_I - old_avg_H_I) < rel_error) and  (np.abs(old_avg_land - avg_temperature_land) < rel_error):
            # We can assume that the error is sufficiently small now.
            verbose and print("Equilibrium reached!")
            if np.abs(avg_H_I - old_avg_H_I) > 0.1 or  np.abs(avg_temperature_atm - old_avg_atm) > 0.1 or np.abs(avg_temperature_s - old_avg_s) > 0.1 or np.abs(avg_temperature_ocn - old_avg_ocn) > 0.1 or np.abs(avg_temperature_land - old_avg_land) > 0.1 :
                Surface_Temp =np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
            break
        
        else:
              old_avg_atm = avg_temperature_atm
              old_avg_ocn = avg_temperature_ocn
              old_avg_s = avg_temperature_s
              old_avg_H_I = avg_H_I
              old_avg_land = avg_temperature_land
  
         
    return  T_ATM, T_S_land, T_S_Ocean, T_OCN, H_I, Surface_Temp, phi_index_s, phi_index_n, Geo_dat, solar_forcing

def tune_equ(x, Aup, Bup, Adn, Bdn):  
    
    # Aup = 5.41689219e+02
    # Bup = 5.90448306e+00
    # Adn = 3.74854527e+02
    # Bdn = 7.13487806e-16
    
    Bolr = 13.56872794
    Aolr = 342.18438355
    
    Albedo_up =0.72
    Albedo_dn =0.25
     
    T_ATM, T_S_land, T_S, T_OCN, H_I, Surface_Temp, phi_index_s, phi_index_n, geo_dat, solar_forcing  = compute_equilibrium( mesh, diffusion_coeff_atm, P_atm.heat_capacity, P_ocn, diffusion_coeff_ocn, heat_capacity_ocn, phi, true_longitude, ntimesteps, Geo_dat0, heat_capacity_s, Ocean_boundary, Lakes, Surface_boundary,  c02_warming, Aup,Bup,Adn,Bdn, Aolr , Bolr ,Albdeo_ERA5, Albedo_up, Albedo_dn)    
    
    VGL_Temp = np.zeros((12,65,128))    
    for t in range(12):
        VGL_Temp[(t+3)%12,:,:] =  (Surface_Temp[:,:, t*4+1] + Surface_Temp[:,:,t*4+2] +Surface_Temp[:,:,t*4+3] +Surface_Temp[:,:,(t*4+4)%48] )/4
         
    avg_nrd = [Functions.calc_mean_north(VGL_Temp[t,:,:], mesh.area) for t in range(12)]
    avg_south = [Functions.calc_mean_south(VGL_Temp[t,:,:], mesh.area) for t in range(12)]
    avg = np.zeros((12,2))
    avg[:,0] = avg_nrd
    avg[:,1] = avg_south
   
    
    return avg.flatten()
    
# Run code
if __name__ == '__main__':

    mesh = Mesh()
    # file_path = '/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/input/The_World128x65.dat.txt'  
    # Geo_dat = Functions.read_geography(file_path)
    test_file = "/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/Version_Paper_2D/netCDF_Data/Era5_land_t2m_celsius_grid_time_mean.nc"      #Era5_land_t2m_celsius_grid_T63_time_mean.nc"   #  #
    data_file = "/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/Version_Paper_2D/netCDF_Data/era5_t2m_world_celsius_grid_T42_year_mean.nc" #era5_t2m_world_celsius_year_mean_Grid_63.nc" #era5_t2m_world_celsius_grid_T42_year_mean.nc"   #
    albedo_file = "/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/Version_Paper_2D/netCDF_Data/albedo_world_year_grid_T42.nc" #albedo_year_mean_Grid_T42.nc"  #albedo_world_time_mean_grid_T42.nc
    
    data, lat , long = Functions.transform_net_cdf(test_file, mesh)
    t2m, lat2, kong2  = Functions.transform_net_cdf(data_file, mesh)
    Albdeo_ERA5, lat3, long3 = Functions.transform_net_cdf(albedo_file, mesh, "fal")
    #Albdeo_ERA5 = np.zeros((12,65,128)) # benutzen wenn nicht in Modell verwendet werden soll

    Geo_dat0 = Functions.Geo_dat_from_NCDF(data, mesh) 
    
    
    # For Lat long Grid T42
    for i in range(mesh.n_latitude): 
        for j in range(mesh.n_longitude):
            if( i > 54 or (i<10 and 39< j< 80)) and Geo_dat0[i,j] == 1: 
                Geo_dat0[i,j] = 3
                
            if 50>i >6 and (39 <j<42 or 56<j<60 or 68<j<80 )and Geo_dat0[i,j] == 3:
                Geo_dat0[i,j] = 1
                
            # if i<=4 and  Geo_dat0[i,j] == 1:
            #     Geo_dat0[i,j] = 3
    
    # #Für Grid T63
    # for i in range(mesh.n_latitude): 
    #     for j in range(mesh.n_longitude):
    #         if( i > 82 or (i<15 and 58< j< 108)) and Geo_dat0[i,j] == 1: 
    #             Geo_dat0[i,j] = 3
                
    #         if 50>i >8 and (58 <j<66 or 102<j<108 or 89<j<102)and Geo_dat0[i,j] == 3:
    #             Geo_dat0[i,j] = 1
                
    #         if (i<7 and 102 <j< 197 and Geo_dat0[i,j] == 1):    
    #             Geo_dat0[i,j] = 3
    
  

    Ocean_boundary = Functions.Get_Ocean_Boundary_Distribution(Geo_dat0)
    Surface_boundary = Functions.Get_Surface_Boundary_Distribution(Geo_dat0)
   # heat_capacity_s = Functions.calc_heat_capacity(Geo_dat)

    Lakes =  Functions.LandDstr_wLakes(Geo_dat0)
    
    ntimesteps = 48
    dt = 1/ ntimesteps
    ecc= 0.016740 #old ecc
    true_longitude = Functions.calc_lambda(dt,  ntimesteps, ecc , per = 1.783037)
    
   
    co2_ppm = 315
    c02_warming = Functions.calc_radiative_cooling_co2(co2_ppm)
  
    
    phi = np.linspace(np.pi/2,-np.pi/2,mesh.n_latitude) #from 90° south to 90° north

    phi_i_deg_n = 75 #inital value for the latitude of the ice-edge 
    phi_i_deg_s = 75 
    
    P_atm = P_atm() #Parameters for the atmosphere
    P_ocn = P_ocn() #Parameters for the ocean
    
    diffusion_coeff_atm = P_atm.heat_capacity * P_atm.diffusion_coeff /mesh.RE**2 #Diffusion coefficient
    
    heat_capacity_ocn = P_ocn.c_O * P_ocn.rhoo * P_ocn.Hml_const * np.ones((mesh.n_latitude, mesh.n_longitude))  # Hml can also be variable
    diffusion_coeff_ocn = heat_capacity_ocn * P_ocn.K_O / mesh.RE**2  #Diffusion coefficient
   
    heat_capacity_s = np.array([mesh.C_s, mesh.C_snow])
    x, y= np.mgrid[0:65,0:128]
    xdata = x.flatten()
    
    
    avg_nrd_t2m = [Functions.calc_mean_north(t2m[t,:,:], mesh.area) for t in range(12)]
    avg_south_t2m = [Functions.calc_mean_south(t2m[t,:,:], mesh.area) for t in range(12)]
    avg_t2m = np.zeros((12,2))
    avg_t2m[:,0] = avg_nrd_t2m
    avg_t2m[:,1] = avg_south_t2m
    # popt, pcov = curve_fit(tune_equ, xdata, avg_t2m.flatten(), p0 = [0.71,0.25], bounds = ([0,0], [ 1,1]),method='dogbox')
    # popt, pcov = curve_fit(tune_equ, xdata, avg_t2m.flatten(), p0 = [mesh.A_up, mesh.B_up, mesh.A_dn, mesh.B_dn], bounds = ([0,0,0,0], [ 10000, 10000, 10000, 10000]))
 
   # tt , Surface_Temp = tune_equ(x,0.5258475862283599 ,0.9423253674935673)

    Bolr = 13.56872794
    Aolr = 342.18438355
    
      
    # Aup = 5.41689219e+02
    # Bup = 5.90448306e+00
    # Adn = 3.74854527e+02
    # Bdn = 7.13487806e-16
    Aup = 257.08540031
    Bup = 9.37315652
    Adn = 299.9406387 
    Bdn =  17.84291327
    
    Albedo_up =0.72
    Albedo_dn =0.25

  
    T_ATM, T_S_land, T_S_Ocean, T_OCN, H_I, Surface_Temp, phi_index_s, phi_index_n, Geo_dat, solar_forcing  = compute_equilibrium( mesh, diffusion_coeff_atm, P_atm.heat_capacity, P_ocn, diffusion_coeff_ocn, heat_capacity_ocn, phi, true_longitude, ntimesteps, Geo_dat0, heat_capacity_s, Ocean_boundary, Lakes, Surface_boundary,  c02_warming, Aup,Bup,Adn,Bdn, Aolr, Bolr, Albdeo_ERA5, Albedo_up, Albedo_dn)    

    
    plt.imshow(Surface_Temp[:,:,0])
    
    annual_mean_temperature_north_ = [Functions.calc_mean_north(Surface_Temp[:, :, t], mesh.area) for t in range(ntimesteps)]
    annual_mean_temperature_south_ = [Functions.calc_mean_south(Surface_Temp[:, :, t], mesh.area) for t in range(ntimesteps)]
    annual_mean_temperature_total_ = [Functions.calc_mean(Surface_Temp[:, :, t], mesh.area) for t in range(ntimesteps)]

    average_temperature_north_ = np.sum(annual_mean_temperature_north_) / ntimesteps
    average_temperature_south_ = np.sum(annual_mean_temperature_south_) / ntimesteps
    average_temperature_total_ = np.sum(annual_mean_temperature_total_) / ntimesteps

    Functions.plot_annual_temperature_north_south(annual_mean_temperature_north_, annual_mean_temperature_south_,
                                        annual_mean_temperature_total_, average_temperature_north_,
                                        average_temperature_south_, average_temperature_total_, "Annual Surface Temperature")
   
    
        # Cologne Plot 
    annual_temperature_cologne = (Surface_Temp[14, 67,:] +Surface_Temp[14, 68, :]) / 2
    #annual_temperature_cologne = (Surface_Temp[21, 100,:])
    #annual_temperature_cologne = (Surface_Temp[20, 101, :] + Surface_Temp[21, 101, :]) / 2
    average_temperature_cologne = np.sum(annual_temperature_cologne) / ntimesteps     
    Functions.plot_temperature_time(annual_temperature_cologne, average_temperature_cologne, "Annual temperature in Cologne")
   
    H_I_nan = H_I
    H_I_nan[H_I_nan == 0] = np.NAN

    annual_mean_H_I_north =[Functions.calc_mean_ocn_north(H_I_nan[:,:,t], mesh.area) for t in range(ntimesteps)]
    Functions.plot_ice_thickness_time(annual_mean_H_I_north, np.mean(annual_mean_H_I_north) , "Mean Ice Thickness North")
    
    annual_mean_H_I_south =[Functions.calc_mean_ocn_south(H_I_nan[:,:,t], mesh.area) for t in range(ntimesteps)]
    Functions.plot_ice_thickness_time(annual_mean_H_I_south, np.mean(annual_mean_H_I_south) , "Mean Ice Thickness South") 
    
    # # Surface Temp 
    # annual_mean_temperature_north_ = [Functions.calc_mean_north(Surface_Temp[:, :, t], mesh.area) for t in range(ntimesteps)]
    # annual_mean_temperature_south_ = [Functions.calc_mean_south(Surface_Temp[:, :, t], mesh.area) for t in range(ntimesteps)]
    # annual_mean_temperature_total_ = [Functions.calc_mean(Surface_Temp[:, :, t], mesh.area) for t in range(ntimesteps)]

    # average_temperature_north_ = np.sum(annual_mean_temperature_north_) / ntimesteps
    # average_temperature_south_ = np.sum(annual_mean_temperature_south_) / ntimesteps
    # average_temperature_total_ = np.sum(annual_mean_temperature_total_) / ntimesteps

    # #ERA5
    
    # annual_mean_temperature_north_ERA5 = [Functions.calc_mean_north(t2m[(t+3)%12,:, :], mesh.area) for t in range(12)]
    # annual_mean_temperature_south_ERA5 = [Functions.calc_mean_south(t2m[(t+3)%12,:, :], mesh.area) for t in range(12)]
    # annual_mean_temperature_total_ERA5 = [Functions.calc_mean(t2m[(t+3)%12,:, :], mesh.area) for t in range(12)]

    # average_temperature_north_ERA5 = np.sum(annual_mean_temperature_north_ERA5) / 12
    # average_temperature_south_ERA5 = np.sum(annual_mean_temperature_south_ERA5) / 12
    # average_temperature_total_ERA5 = np.sum(annual_mean_temperature_total_ERA5) / 12
   
    
    # annual_mean_temperature_north_12 = np.zeros((12))    
    # annual_mean_temperature_south_12 = np.zeros((12))    
    # annual_mean_temperature_total_12 = np.zeros((12))    
    # for t in range(12):
    #     annual_mean_temperature_north_12[(t+3)%12] =  (annual_mean_temperature_north_[t*4+1] + annual_mean_temperature_north_[t*4+2] +annual_mean_temperature_north_[t*4+3] +annual_mean_temperature_north_[(t*4+4)%48] )/4
    #     annual_mean_temperature_south_12[(t+3)%12] =  (annual_mean_temperature_south_[t*4+1] + annual_mean_temperature_south_[t*4+2] +annual_mean_temperature_south_[t*4+3] +annual_mean_temperature_south_[(t*4+4)%48] )/4
    #     annual_mean_temperature_total_12[(t+3)%12] =  (annual_mean_temperature_total_[t*4+1] + annual_mean_temperature_total_[t*4+2] +annual_mean_temperature_total_[t*4+3] +annual_mean_temperature_total_[(t*4+4)%48] )/4

                                      
    # fig, ax = plt.subplots()
    # ntimesteps = len(annual_mean_temperature_total_ERA5)
    # plt.plot(average_temperature_total_ERA5 * np.ones(ntimesteps), label="average temperature (total) ERA5")
    # plt.plot(annual_mean_temperature_total_ERA5, label="temperature (total) ERA5")
    # plt.plot(average_temperature_total_ * np.ones(ntimesteps), label="average temperature (total)")
    # plt.plot(annual_mean_temperature_total_12, label="temperature (total)")

    # plt.xlim((0, ntimesteps - 1))
    # labels = ["March", "June", "September", "December", "March"]
    # plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    # ax.set_ylabel("surface temperature [°C]")
    # plt.grid()
    # plt.title("Annual Surface Temperature")
    # plt.legend(loc="upper right")

    # plt.tight_layout()
    # plt.show()
    
    # plt.plot(average_temperature_north_ERA5 * np.ones(ntimesteps), label="average temperature (north) ERA5")
    # plt.plot(annual_mean_temperature_north_ERA5, label="temperature (north) ERA5")
    # plt.plot(average_temperature_north_ * np.ones(ntimesteps), label="average temperature (north)")
    # plt.plot(annual_mean_temperature_north_12, label="temperature (north)")
   
    # plt.xlim((0, ntimesteps - 1))
    # labels = ["March", "June", "September", "December", "March"]
    # plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    # ax.set_ylabel("surface temperature [°C]")
    # plt.grid()
    # plt.title("Annual Surface Temperature")
    # plt.legend(loc="upper right")

    # plt.tight_layout()
    # plt.show()
    
    
    # plt.plot(average_temperature_south_ERA5 * np.ones(ntimesteps), label="average temperature (south) ERA5")
    # plt.plot(annual_mean_temperature_south_ERA5, label="temperature (south) ERA5")
    # plt.plot(average_temperature_south_ * np.ones(ntimesteps), label="average temperature (south)")
    # plt.plot(annual_mean_temperature_south_12, label="temperature (south)")
 
 
    # plt.xlim((0, ntimesteps - 1))
    # labels = ["March", "June", "September", "December", "March"]
    # plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    # ax.set_ylabel("surface temperature [°C]")
    # plt.grid()
    # plt.title("Annual Surface Temperature")
    # plt.legend(loc="upper right")

    # plt.tight_layout()
    # plt.show()
    