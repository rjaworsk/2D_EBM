#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:08:23 2023

@author: ricij

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 08:51:38 2023

@author: ricij
"""
import copy
import numpy as np
import time
import xarray as xr
from scipy import sparse
import scipy.sparse.linalg as lg
from scipy.optimize import curve_fit

import Functions
import ATM_fct
import OCN_fct
import Surface_fct
from scipy.optimize import least_squares

class Mesh():
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
  

def surface_temp_land(T_ATM, solar_forcing, phi, mesh, Geo_dat, T_S_land, delta_t, X1,X2,X3,X4, T_S_zero, t, data, i): #calc surface temp of land without diffusion
    
    # T_S_land_r = T_S_land.flatten()
    
    #RHS = 1/heat_capacity * (solar_forcing - 2.15 *T_S_land - 210.3) # From lecture
    #RHS = 1/mesh.C_s * (solar_forcing + mesh.B_dn * T_ATM - mesh.A_up  + mesh.A_dn - mesh.B_up * T_S_land )
    
    # def RHS(T_S_land_new,X1,X2,X3,X4):
    #   RHS_ =  1/mesh.C_s * (solar_forcing[:,:,t-1].flatten() + X1 * T_ATM[:,:,t-1].flatten() - X4  + X3 - X2* T_S_land[:,:,t-1].flatten() )
    #   T_S_land_new = T_S_land[:,:,t-1].flatten() + delta_t * RHS_
    #   return  T_S_land_new
    
    # if t == 38 or t == 39  or t == 40 or t == 41: 
    #     T_S_zero = copy.copy(data[0,:,:])
    #     T_S_zero[np.isnan(T_S_zero)] = 0
    # elif  t == 42 or t == 43  or t ==  44 or t == 45: 
    #     T_S_zero = copy.copy(data[1,:,:])
    #     T_S_zero[np.isnan(T_S_zero)] = 0
    # elif  t == 46 or t == 47  or t == 48 or t == 1: 
    #     T_S_zero = copy.copy(data[2,:,:])
    #     T_S_zero[np.isnan(T_S_zero)] = 0
    # elif  t == 2 or t == 3  or t == 4 or t == 5: 
    #     T_S_zero = copy.copy(data[3,:,:])
    #     T_S_zero[np.isnan(T_S_zero)] = 0
    # elif  t == 6 or t == 7  or t == 8 or t == 9: 
    #     T_S_zero = copy.copy(data[4,:,:])
    #     T_S_zero[np.isnan(T_S_zero)] = 0
    # elif  t == 10 or t == 11  or t == 12 or t == 13: 
    #     T_S_zero = copy.copy(data[5,:,:])
    #     T_S_zero[np.isnan(T_S_zero)] = 0
    # elif  t == 14 or t == 15  or t == 16 or t == 17: 
    #     T_S_zero = copy.copy(data[6,:,:])
    #     T_S_zero[np.isnan(T_S_zero)] = 0
    # elif  t == 18 or t == 19  or t == 20 or t == 21: 
    #     T_S_zero = copy.copy(data[7,:,:])
    #     T_S_zero[np.isnan(T_S_zero)] = 0
    # elif  t == 22 or t == 23  or t == 24 or t == 25: 
    #     T_S_zero = copy.copy(data[8,:,:])
    #     T_S_zero[np.isnan(T_S_zero)] = 0
    # elif  t == 26 or t == 27  or t == 28 or t == 29: 
    #     T_S_zero = copy.copy(data[9,:,:])
    #     T_S_zero[np.isnan(T_S_zero)] = 0
    # elif  t == 30 or t == 31  or t == 32 or t == 33: 
    #     T_S_zero = copy.copy(data[10,:,:])
    #     T_S_zero[np.isnan(T_S_zero)] = 0
    # elif  t == 34 or t == 35  or t == 36 or t ==37: 
    #     T_S_zero = copy.copy(data[12,:,:])
    #     T_S_zero[np.isnan(T_S_zero)] = 0  
    # T_fit = np.zeros((65,128,48))    
    # for t in range(48): 
    #     if t == 38 or t == 39  or t == 40 or t == 41: 
    #         T_S_zero = copy.copy(data[0,:,:])
    #         T_S_zero[np.isnan(T_S_zero)] = 0
    #         T_fit[:,:,t] = T_S_zero
    #     elif  t == 42 or t == 43  or t ==  44 or t == 45: 
    #         T_S_zero = copy.copy(data[1,:,:])
    #         T_S_zero[np.isnan(T_S_zero)] = 0
    #         T_fit[:,:,t] = T_S_zero
    #     elif  t == 46 or t == 47  or t == 48 or t == 1: 
    #         T_S_zero = copy.copy(data[2,:,:])
    #         T_S_zero[np.isnan(T_S_zero)] = 0
    #         T_fit[:,:,t] = T_S_zero
    #     elif  t == 2 or t == 3  or t == 4 or t == 5: 
    #         T_S_zero = copy.copy(data[3,:,:])
    #         T_S_zero[np.isnan(T_S_zero)] = 0
    #         T_fit[:,:,t] = T_S_zero
    #     elif  t == 6 or t == 7  or t == 8 or t == 9: 
    #         T_S_zero = copy.copy(data[4,:,:])
    #         T_S_zero[np.isnan(T_S_zero)] = 0
    #         T_fit[:,:,t] = T_S_zero
    #     elif  t == 10 or t == 11  or t == 12 or t == 13: 
    #         T_S_zero = copy.copy(data[5,:,:])
    #         T_S_zero[np.isnan(T_S_zero)] = 0
    #         T_fit[:,:,t] = T_S_zero
    #     elif  t == 14 or t == 15  or t == 16 or t == 17: 
    #         T_S_zero = copy.copy(data[6,:,:])
    #         T_S_zero[np.isnan(T_S_zero)] = 0
    #         T_fit[:,:,t] = T_S_zero
    #     elif  t == 18 or t == 19  or t == 20 or t == 21: 
    #         T_S_zero = copy.copy(data[7,:,:])
    #         T_S_zero[np.isnan(T_S_zero)] = 0
    #         T_fit[:,:,t] = T_S_zero
    #     elif  t == 22 or t == 23  or t == 24 or t == 25: 
    #         T_S_zero = copy.copy(data[8,:,:])
    #         T_S_zero[np.isnan(T_S_zero)] = 0
    #         T_fit[:,:,t] = T_S_zero
    #     elif  t == 26 or t == 27  or t == 28 or t == 29: 
    #         T_S_zero = copy.copy(data[9,:,:])
    #         T_S_zero[np.isnan(T_S_zero)] = 0
    #         T_fit[:,:,t] = T_S_zero
    #     elif  t == 30 or t == 31  or t == 32 or t == 33: 
    #         T_S_zero = copy.copy(data[10,:,:])
    #         T_S_zero[np.isnan(T_S_zero)] = 0
    #         T_fit[:,:,t] = T_S_zero
    #     elif  t == 34 or t == 35  or t == 36 or t ==37: 
    #         T_S_zero = copy.copy(data[12,:,:])
    #         T_S_zero[np.isnan(T_S_zero)] = 0
    #         T_fit[:,:,t] = T_S_zero
    
    #if i == 0:    
    # x, y= np.mgrid[0:65,0:128]
    # xdata = x.flatten()
    # popt, pcov = curve_fit(RHS, xdata, T_S_zero.flatten(), p0 = [mesh.B_dn, mesh.A_up, mesh.A_dn, mesh.B_up])
    # X1 =  popt[0]
    # X2 =  popt[1]
    # X3 =  popt[2]
    # X4 =  popt[3]
    RHS_e =  1/mesh.C_s * (solar_forcing[:,:,t-1] + X1 * T_ATM[:,:,t-1] - X4  + X3 - X2* T_S_land[:,:,t-1] )
    T_S_land_new = T_S_land[:,:,t-1] + delta_t * RHS_e  
    
    for i in range(mesh.n_latitude):
        for j in range(mesh.n_longitude):
              if Geo_dat[i,j] == 5: 
                  T_S_land_new[i,j] = 0
   
    return T_S_land_new, X1, X2, X3, X4

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

def FreezeAndMelt(T_OCN, H_I, Hml, mesh, P_ocn): #calc new ice distribution
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


def Snow_melt(Geo_dat, T_ATM,T_S_land ,mesh): # Calculate if there is snow present from a given surface temperature
    
    def Model(t2m): 
        return 1.54425e-14 - 0.00577 * t2m

    pred = Model(T_S_land)

    for i in range(mesh.n_latitude):  
       for j in range(mesh.n_longitude):
               if pred[i,j] >0.001 and Geo_dat[i,j] == 1: 
                   Geo_dat[i,j] = 3
               elif pred[i,j] <= 0.001 and Geo_dat[i,j] == 3: 
                   Geo_dat[i,j] = 1
                        
    return Geo_dat  
              
def compute_equilibrium(mesh, diffusion_coeff_atm, heat_capacity_atm, P_ocn, diffusion_coeff_ocn, heat_capacity_ocn,  phi, true_longitude,n_timesteps, Geo_dat, heat_capacity_s, diffusion_coeff_s, Ocean_boundary, Lakes, Surface_boundary,c02_warming, X1,X2,X3,X4, T_S_zero, data, max_iterations=50, rel_error=1e-5, verbose=True):
    # Step size
    delta_t = 1 / 48
    ntimesteps = 48
    
    #Inital Conditions
    T_ATM = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    T_OCN = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    T_S_Ocean = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    T_S_land = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    H_I = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    Surface_Temp = np.zeros((mesh.n_latitude,mesh.n_longitude, ntimesteps))
   
    
    # Area-mean in every time step
    temp_atm = np.zeros(ntimesteps)
    temp_ocn=  np.zeros(ntimesteps)
    temp_s =  np.zeros(ntimesteps)
    temp_H_I =  np.zeros(ntimesteps)
    temp_land = np.zeros(ntimesteps)
    
    phi_index_s = np.zeros((mesh.n_longitude,ntimesteps))
    phi_index_n = np.zeros((mesh.n_longitude,ntimesteps))
    solar_forcing = np.zeros((mesh.n_latitude, mesh.n_longitude,ntimesteps))
    

    # Average temperature over all time steps from the previous iteration to approximate the error
    old_avg_atm = 0
    old_avg_ocn = 0
    old_avg_s = 0
    old_avg_H_I = 0
    old_avg_land = 0
    
    Fb = OCN_fct.BasalFlux(phi, mesh) #additional ocean flux
    Hml = P_ocn.Hml_const * np.ones((mesh.n_latitude,mesh.n_longitude)) # Can be changed 
    
    # Construct and factorize Jacobian for the atmosphere
    jacobian_atm = ATM_fct.calc_jacobian_atm(mesh, diffusion_coeff_atm, P_atm.heat_capacity, phi, Geo_dat)
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
    
   
    # Construct and factorize Jacobian for the land
    # jacobian_s = Surface_fct.calc_jacobian_s(mesh, diffusion_coeff_s, heat_capacity_s, phi, Surface_boundary)
    # m, n = jacobian_s.shape
    # eye = sparse.eye(m, n, format="csc")
    # jacobian_s = sparse.csc_matrix(jacobian_s)
    # jacobian_s = lg.factorized(eye - delta_t * jacobian_s)
    
    # Compute insolation
    insolation = Functions.calc_insolation(phi, true_longitude)
    
    
    timer = Timer()
    for i in range(max_iterations):
        print("Iteration: ",i)
        timer.start()
        for t in range(ntimesteps):    
            
            phi_index_s[:,t], phi_i_s, phi_index_n[:,t], phi_i_n = ice_edge(H_I[:,:,t-1], phi, Geo_dat)  # new Ice_Edge Index 
            
            Geo_dat = Functions.change_geo_dat(Geo_dat, H_I[:,:,t-1], mesh) #calc new ice distribution
            
            Geo_dat = Snow_melt(Geo_dat, T_ATM[:,:,t-1], Surface_Temp[:,:,t-1], mesh) #new snow distribution
            
            coalbedo = Functions.calc_coalbedo(Geo_dat, phi_i_n, phi_i_s) #new coalbdeo dependant on the ice_edge 
            
            solar_forcing[:,:,t-1]  = Functions.calc_solar_forcing(insolation[:,t-1],coalbedo, mesh)        
            
            T_ATM[:,:,t] = ATM_fct.timestep_euler_backward_atm(jacobian_atm, 1/ntimesteps, T_ATM[:,:,t-1], Surface_Temp[:,:,t-1], mesh, P_atm.heat_capacity, c02_warming, Geo_dat)
            
            T_OCN[:,:,t] = OCN_fct.timestep_euler_backward_ocn(jacobian_ocn, 1/ntimesteps, T_OCN[:,:,t-1], T_S_Ocean[:,:,t-1], T_ATM[:,:,t-1], mesh, heat_capacity_ocn, solar_forcing[:,:,t-1], Fb, H_I[:,:,t-1], Geo_dat, Lakes)
            
            
            H_I[:,:,t] = timestep_euler_forward_H_I(mesh,T_S_Ocean[:,:,t-1], T_ATM[:,:,t-1], Fb, solar_forcing[:,:,t-1], H_I[:,:,t-1], t, delta_t)
             
            T_OCN[:,:,t], H_I[:,:,t] = FreezeAndMelt(T_OCN[:,:,t], H_I[:,:,t], Hml, mesh, P_ocn) 
             
            #T_S_land[:,:,t] = Surface_fct.timestep_euler_backward_s(jacobian_s, 1/ntimesteps, T_S_land[:,:,t-1], T_ATM[:,:,t-1], t, mesh, solar_forcing, Geo_dat,  heat_capacity_s) #with diffusion
    
            T_S_land[:,:,t],X1, X2, x3, X4 = surface_temp_land(T_ATM, solar_forcing, phi, mesh, Geo_dat, T_S_land, delta_t, X1,X2,X3,X4, T_S_zero, t, data, i) #without diffusion
            
            solar_forcing_new = Functions.calc_solar_forcing(insolation[:,t],coalbedo, mesh)   
           
            T_S_Ocean[:,:,t] = surface_temp_ocean(T_ATM[:,:,t], T_OCN[:,:,t], H_I[:,:,t], solar_forcing_new, phi, mesh, Geo_dat) #original surface temp from Ocean Model --> is just for the ocean surface temperatue
           
            Surface_Temp[:,:,t] = Functions.unite_surface_temp(T_S_Ocean[:,:,t] , T_S_land[:,:,t] , mesh, Geo_dat)
            
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
        print("X1: ",X1, " X2: ", X2, " X3: ", X3, " X4: ",X4)
        if (np.abs(avg_temperature_atm - old_avg_atm)< rel_error) and (np.abs(avg_temperature_ocn - old_avg_ocn)< rel_error)  and  (np.abs(avg_temperature_s - old_avg_s)< rel_error) and (np.abs(avg_H_I - old_avg_H_I) < rel_error):
            # We can assume that the error is sufficiently small now.
            verbose and print("Equilibrium reached!")
            
            break
        
        else:
              old_avg_atm = avg_temperature_atm
              old_avg_ocn = avg_temperature_ocn
              old_avg_s = avg_temperature_s
              old_avg_H_I = avg_H_I
              old_avg_land = avg_temperature_land
  
         
    return  T_ATM, T_S_land, T_S_Ocean, T_OCN, H_I, Surface_Temp, phi_index_s, phi_index_n, X1, X2, X3, X4
       

def equ_test(T_S_land,X1,X2,X3,X4):
    
    T_ATM, T_S_land, T_S, T_OCN, H_I, Surface_Temp, phi_index_s, phi_index_n, X1, X2, X3, X4  = compute_equilibrium( mesh, diffusion_coeff_atm, P_atm.heat_capacity, P_ocn, diffusion_coeff_ocn, heat_capacity_ocn, phi, true_longitude, ntimesteps, Geo_dat, heat_capacity_s, diffusion_coeff_s, Ocean_boundary, Lakes, Surface_boundary, c02_warming,X1,X2,X3,X4,T_S_zero, data )
   
    return T_S_land.flatten()

# Run code
if __name__ == '__main__':
    # start = time.time()  
   # file_path = '/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/input/The_World128x65.dat.txt'  
    #Geo_dat = Functions.read_geography(file_path)
    #Geo_dat = Functions.get_data_with_new_resolution(Geo_dat_old, 129)

    file_path = '/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/input/The_World128x65.dat.txt'  
    mesh = Mesh()
    #Geo_dat = Functions.get_data_with_new_resolution(Geo_dat_old, 129)
    test_file = "/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/Version_Paper_2D/netCDF_Data/Era5_land_t2m_celsius_grid.nc"    
    data, lat , long = Functions.transform_net_cdf(test_file,mesh)
    Geo_dat = Functions.Geo_dat_from_NCDF(data, mesh)
  
    Ocean_boundary = Functions.Get_Ocean_Boundary_Distribution(Geo_dat)
    Surface_boundary = Functions.Get_Surface_Boundary_Distribution(Geo_dat)

    mesh = Mesh()
     
    heat_capacity_s = Functions.calc_heat_capacity(Geo_dat)
   # diffusion_coeff_s = Functions.calc_diffusion_coefficients(Geo_dat)  #noch nicht eingebaut
    diffusion_coeff_s = np.ones((mesh.n_latitude, mesh.n_longitude)) * 0.18
    # for i in range(mesh.n_latitude):
    #     for j in range(mesh.n_longitude):
    #           if Geo_dat[i,j] == 5 or  Geo_dat[i,j] == 2:
    #               diffusion_coeff_s[i,j] = 0
    
    Lakes =  Functions.LandDstr_wLakes(Geo_dat)
    
    ntimesteps = 48
    dt = 1/ ntimesteps
    ecc= 0.016740 #old ecc
    true_longitude = Functions.calc_lambda(dt,  ntimesteps, ecc =  0.016740, per = 1.783037)
    
   
    co2_ppm = 315.0
    c02_warming = Functions.calc_radiative_cooling_co2(co2_ppm)
  
    
    phi = np.linspace(np.pi/2,-np.pi/2,mesh.n_latitude) #from 90° south to 90° north

    phi_i_deg_n = 75 #inital value for the latzitude of the ice-edge 
    phi_i_deg_s = 75 
    
    P_atm = P_atm() #Parameters for the atmosphere
    P_ocn = P_ocn() #Parameters for the ocean
    
    diffusion_coeff_atm = P_atm.heat_capacity * P_atm.diffusion_coeff /mesh.RE**2 #Diffusion coefficient
    
    heat_capacity_ocn = P_ocn.c_O * P_ocn.rhoo * P_ocn.Hml_const * np.ones((mesh.n_latitude, mesh.n_longitude))  # Hml can also be variable
    diffusion_coeff_ocn = heat_capacity_ocn * P_ocn.K_O / mesh.RE**2  #Diffusion coefficient
   
    X1 = mesh.B_dn
    X2 = mesh.B_up 
    X3 = mesh.A_dn 
    X4 = mesh.A_up  #TEST
    T_S_zero = copy.copy(data)
    T_S_zero[np.isnan(T_S_zero)] = 0
    
   
    x, y= np.mgrid[0:65,0:128]
    xdata = np.vstack((x.ravel(), y.ravel()))
    param_bounds=[(0,None),(None,0),(0,None),(None,0)] 
    popt, pcov = curve_fit(equ_test, xdata, T_S_zero.flatten (),bounds=([0,-np.inf,0,-np.inf], [np.inf, 0, np.inf,0]) )
    
   
                    
    ###### run model ######
    # T_ATM, T_S_land, T_S, T_OCN, H_I, Surface_Temp, phi_index_s, phi_index_n, X1, X2, X3, X4  = compute_equilibrium( mesh, diffusion_coeff_atm, P_atm.heat_capacity, P_ocn, diffusion_coeff_ocn, heat_capacity_ocn, phi, true_longitude, ntimesteps, Geo_dat, heat_capacity_s, diffusion_coeff_s, Ocean_boundary, Lakes, Surface_boundary, c02_warming,X1,X2,X3,X4,T_S_zero, data )
   
    # end = time.time()
    # print(end - start)
 