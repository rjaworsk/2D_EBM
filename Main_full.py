#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:16:38 2023

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
from scipy import sparse
import scipy.sparse.linalg as lg

import Functions
import ATM_fct
import OCN_fct
import Surface_fct

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
  

def surface_temp_land(T_ATM, solar_forcing, phi, mesh, Geo_dat, T_S_land, delta_t, heat_capacity): #calc surface temp of land without diffusion
    T_S_land_new = np.zeros((mesh.n_latitude, mesh.n_longitude))
    RHS = np.zeros((mesh.n_latitude, mesh.n_longitude))
   
   # RHS = 1/mesh.C_s * (solar_forcing +  6.04922 * T_ATM - 386.92082  + 334.6069 - 10.6018 * T_S_land) # Tuning mit Least Squares
   # RHS = 1/mesh.C_s * (solar_forcing + 6.5 * T_ATM - mesh.A_up  + mesh.A_dn - 10 * T_S_land) #selbst getuned
    #RHS = 1/0.27827695 * (solar_forcing + mesh.B_dn * T_ATM - mesh.A_up  + mesh.A_dn - mesh.B_up * T_S_land) #og
   # RHS = 1/mesh.C_s * (solar_forcing + 2.62776805e-14 * T_ATM - 489.543109  + 328.30824  - 6.71741559 * T_S_land) 
    #RHS = 1/mesh.C_s * (solar_forcing  - 4.3654310 * T_ATM - 691.7362  + 427.022 - 6.6669 * T_S_land)
    
    for i in range(mesh.n_latitude):
        for j in range(mesh.n_longitude):
            if Geo_dat[i,j] == 3:
                RHS[i,j] = 1/heat_capacity[0] * (solar_forcing[i,j]  + mesh.B_dn * T_ATM[i,j] - mesh.A_up  + mesh.A_dn - mesh.B_up * T_S_land[i,j]) #og
            elif Geo_dat[i,j] == 1:
                RHS[i,j] = 1/heat_capacity[1] * (solar_forcing[i,j]  + mesh.B_dn * T_ATM[i,j] - mesh.A_up  + mesh.A_dn - mesh.B_up * T_S_land[i,j]) #og
            else: 
                RHS[i,j] = 0
                
    T_S_land_new = T_S_land + delta_t * RHS
    
    # for i in range(mesh.n_latitude):
    #     for j in range(mesh.n_longitude):
    #         if Geo_dat[i,j] == 5 or Geo_dat[i,j] == 2: 
    #             T_S_land_new[i,j] = 0
            
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


def Snow_melt(Geo_dat, T_ATM,T_S_land ,mesh, Snow): # Calculate if there is snow present from a given surface temperature    
                        
    def Model(t2m, Snow): 
        return Snow + 1.54425e-14 - 0.00577 * t2m

    pred = Model(T_S_land, Snow)

    for i in range(mesh.n_latitude):  
       for j in range(mesh.n_longitude):
               if pred[i,j] >0.001 and Geo_dat[i,j] == 1:  #0.001
                   Geo_dat[i,j] = 3
               elif pred[i,j] <= 0.001 and Geo_dat[i,j] == 3: 
                   Geo_dat[i,j] = 1
                        
    return Geo_dat, pred
              
def compute_equilibrium(mesh, diffusion_coeff_atm, heat_capacity_atm, P_ocn, diffusion_coeff_ocn, heat_capacity_ocn,  phi, true_longitude,n_timesteps, Geo_dat0, heat_capacity_s, diffusion_coeff_s, Ocean_boundary, Lakes, Surface_boundary,  c02_warming, max_iterations=10, rel_error=1e-5, verbose=True):
    # Step size
    delta_t = 1 / ntimesteps
    
    #Inital Conditions
    T_ATM = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    T_OCN = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    T_S_Ocean = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    T_S_land = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    H_I = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    Surface_Temp = np.zeros((mesh.n_latitude,mesh.n_longitude, ntimesteps))
    solar_forcing = np.zeros((mesh.n_latitude,mesh.n_longitude, ntimesteps))
   
   
    
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

    # Average temperature over all time steps from the previous iteration to approximate the error
    old_avg_atm = 0
    old_avg_ocn = 0
    old_avg_s = 0
    old_avg_H_I = 0
    old_avg_land = 0
    
    Fb = OCN_fct.BasalFlux(phi, mesh) #additional ocean flux
    Hml = P_ocn.Hml_const * np.ones((mesh.n_latitude,mesh.n_longitude)) # Can be changed 
    
    # Construct and factorize Jacobian for the atmosphere
    jacobian_atm = ATM_fct.calc_jacobian_atm(mesh, diffusion_coeff_atm, P_atm.heat_capacity, phi, Geo_dat0)
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
            
            phi_index_s[:,t], phi_i_s, phi_index_n[:,t], phi_i_n = ice_edge(H_I[:,:,t-1], phi, Geo_dat[:,:,t-1])  # new Ice_Edge Index 
            
            Geo_dat[:,:,t] = Functions.change_geo_dat(Geo_dat[:,:,t-1], H_I[:,:,t-1], mesh) #calc new ice distribution
            
           # Geo_dat[:,:,t], Snow[:,:,t] = Snow_melt(Geo_dat[:,:,t], T_ATM[:,:,t-1], Surface_Temp[:,:,t-1], mesh, Snow[:,:,t-1]) #new snow distribution
            
            coalbedo = Functions.calc_coalbedo(Geo_dat[:,:,t], phi_i_n, phi_i_s) #new coalbdeo dependant on the ice_edge 
            
            solar_forcing[:,:,t-1]  = Functions.calc_solar_forcing(insolation[:,t-1],coalbedo, mesh)        
            
            T_ATM[:,:,t] = ATM_fct.timestep_euler_backward_atm(jacobian_atm, 1/ntimesteps, T_ATM[:,:,t-1], Surface_Temp[:,:,t-1], mesh, P_atm.heat_capacity, c02_warming, Geo_dat[:,:,t])
            
            T_OCN[:,:,t] = OCN_fct.timestep_euler_backward_ocn(jacobian_ocn, 1/ntimesteps, T_OCN[:,:,t-1], T_S_Ocean[:,:,t-1], T_ATM[:,:,t-1], mesh, heat_capacity_ocn, solar_forcing[:,:,t-1], Fb, H_I[:,:,t-1], Geo_dat[:,:,t], Lakes)
                       
            H_I[:,:,t] = timestep_euler_forward_H_I(mesh,T_S_Ocean[:,:,t-1], T_ATM[:,:,t-1], Fb, solar_forcing[:,:,t-1], H_I[:,:,t-1], t, delta_t)
             
            T_OCN[:,:,t], H_I[:,:,t] = FreezeAndMelt(T_OCN[:,:,t], H_I[:,:,t], Hml, mesh) 
             
            #T_S_land[:,:,t] = Surface_fct.timestep_euler_backward_s(jacobian_s, 1/ntimesteps, T_S_land[:,:,t-1], T_ATM[:,:,t-1], t, mesh, solar_forcing[:,:,t-1], Geo_dat[:,:,t],  heat_capacity_s) #with diffusion
    
            T_S_land[:,:,t] = surface_temp_land(T_ATM[:,:,t], solar_forcing[:,:,t-1], phi, mesh, Geo_dat[:,:,t], T_S_land[:,:,t-1], delta_t, heat_capacity_s) #without diffusion
            
            solar_forcing_new = Functions.calc_solar_forcing(insolation[:,t],coalbedo, mesh)   
           
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
  
         
    return  T_ATM, T_S_land, T_S_Ocean, T_OCN, H_I, Surface_Temp, phi_index_s, phi_index_n, Geo_dat, solar_forcing
       
# Run code
if __name__ == '__main__':
    start = time.time() 
    mesh = Mesh()
    file_path = '/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/input/The_World128x65.dat.txt'  
    Geo_dat = Functions.read_geography(file_path)
    #Geo_dat = Functions.get_data_with_new_resolution(Geo_dat_old, 129)
    test_file = "/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/Version_Paper_2D/netCDF_Data/Era5_land_t2m_celsius_grid.nc"    
    data, lat , long = Functions.transform_net_cdf(test_file, mesh)
    #T_S_land_avg_ = [Functions.calc_mean_ocn(data[t, :], mesh.area) for t in range(48)] #da Daten im Januar starten
    
    #Geo_dat = Functions.Geo_dat_from_NCDF(data, mesh) 

    
    Ocean_boundary = Functions.Get_Ocean_Boundary_Distribution(Geo_dat)
    Surface_boundary = Functions.Get_Surface_Boundary_Distribution(Geo_dat)
   
     
    #heat_capacity_s = Functions.calc_heat_capacity(Geo_dat)
   # diffusion_coeff_s = Functions.calc_diffusion_coefficients(Geo_dat)  #noch nicht eingebaut
    diffusion_coeff_s = np.ones((mesh.n_latitude, mesh.n_longitude)) * 0.18
    
    heat_capacity_s = np.array([mesh.C_snow,mesh.C_s]) #erst Schnee dann Land
    
    Lakes =  Functions.LandDstr_wLakes(Geo_dat)
    
    ntimesteps = 48
    dt = 1/ ntimesteps
    ecc= 0.016740 #old ecc
    true_longitude = Functions.calc_lambda(dt,  ntimesteps, ecc =  0.016740, per = 1.783037)
    
   
    co2_ppm = 388.91 
    c02_warming = Functions.calc_radiative_cooling_co2(co2_ppm)
  
    
    phi = np.linspace(np.pi/2,-np.pi/2,mesh.n_latitude) #from 90° south to 90° north

    phi_i_deg_n = 75 #inital value for the latitude of the ice-edge 
    phi_i_deg_s = 75 
    
    P_atm = P_atm() #Parameters for the atmosphere
    P_ocn = P_ocn() #Parameters for the ocean
    
    diffusion_coeff_atm = P_atm.heat_capacity * P_atm.diffusion_coeff /mesh.RE**2 #Diffusion coefficient
    
    heat_capacity_ocn = P_ocn.c_O * P_ocn.rhoo * P_ocn.Hml_const * np.ones((mesh.n_latitude, mesh.n_longitude))  # Hml can also be variable
    diffusion_coeff_ocn = heat_capacity_ocn * P_ocn.K_O / mesh.RE**2  #Diffusion coefficient
   
                    
    ###### run model ######
    T_ATM, T_S_land, T_S_Ocean, T_OCN, H_I, Surface_Temp, phi_index_s, phi_index_n, Geo_dat, solar_forcing  = compute_equilibrium( mesh, diffusion_coeff_atm, P_atm.heat_capacity, P_ocn, diffusion_coeff_ocn, heat_capacity_ocn, phi, true_longitude, ntimesteps, Geo_dat, heat_capacity_s, diffusion_coeff_s, Ocean_boundary, Lakes, Surface_boundary,  c02_warming)

    end = time.time()
    print(end - start)
    
    ##################################################################################################################################
    ############################################################## PLOTS #############################################################
    ##################################################################################################################################
    
    annual_mean_temperature_north_ = [Functions.calc_mean_north(Surface_Temp[:, :, t], mesh.area) for t in range(ntimesteps)]
    annual_mean_temperature_south_ = [Functions.calc_mean_south(Surface_Temp[:, :, t], mesh.area) for t in range(ntimesteps)]
    annual_mean_temperature_total_ = [Functions.calc_mean(Surface_Temp[:, :, t], mesh.area) for t in range(ntimesteps)]

    average_temperature_north_ = np.sum(annual_mean_temperature_north_) / ntimesteps
    average_temperature_south_ = np.sum(annual_mean_temperature_south_) / ntimesteps
    average_temperature_total_ = np.sum(annual_mean_temperature_total_) / ntimesteps

    Functions.plot_annual_temperature_north_south(annual_mean_temperature_north_, annual_mean_temperature_south_,
                                        annual_mean_temperature_total_, average_temperature_north_,
                                        average_temperature_south_, average_temperature_total_, "Annual Surface Temperature")
    
    annual_mean_temperature_north_ = [Functions.calc_mean_north(T_ATM[:, :, t], mesh.area) for t in range(ntimesteps)]
    annual_mean_temperature_south_ = [Functions.calc_mean_south(T_ATM[:, :, t], mesh.area) for t in range(ntimesteps)]
    annual_mean_temperature_total_ = [Functions.calc_mean(T_ATM[:, :, t], mesh.area) for t in range(ntimesteps)]

    average_temperature_north_ = np.sum(annual_mean_temperature_north_) / ntimesteps
    average_temperature_south_ = np.sum(annual_mean_temperature_south_) / ntimesteps
    average_temperature_total_ = np.sum(annual_mean_temperature_total_) / ntimesteps

    Functions.plot_annual_temperature_north_south(annual_mean_temperature_north_, annual_mean_temperature_south_,
                                        annual_mean_temperature_total_, average_temperature_north_,
                                        average_temperature_south_, average_temperature_total_, "Annual Atmosphere Temperature")
    
   
    annual_mean_temperature_north_ = [Functions.calc_mean_ocn_north(T_OCN[:, :, t], mesh.area) for t in range(ntimesteps)]
    annual_mean_temperature_south_ = [Functions.calc_mean_ocn_south(T_OCN[:, :, t], mesh.area) for t in range(ntimesteps)]
    annual_mean_temperature_total_ = [Functions.calc_mean_ocn(T_OCN[:, :, t], mesh.area) for t in range(ntimesteps)]

    average_temperature_north_ = np.sum(annual_mean_temperature_north_) / ntimesteps
    average_temperature_south_ = np.sum(annual_mean_temperature_south_) / ntimesteps
    average_temperature_total_ = np.sum(annual_mean_temperature_total_) / ntimesteps

    Functions.plot_annual_temperature_north_south(annual_mean_temperature_north_, annual_mean_temperature_south_,
                                        annual_mean_temperature_total_, average_temperature_north_,
                                        average_temperature_south_, average_temperature_total_, "Annual Ocean Temperature")
    
    
    annual_mean_temperature_north_ = [Functions.calc_mean_ocn_north(T_S_Ocean[:, :, t], mesh.area) for t in range(ntimesteps)]
    annual_mean_temperature_south_ = [Functions.calc_mean_ocn_south(T_S_Ocean[:, :, t], mesh.area) for t in range(ntimesteps)]
    annual_mean_temperature_total_ = [Functions.calc_mean_ocn(T_S_Ocean[:, :, t], mesh.area) for t in range(ntimesteps)]

    average_temperature_north_ = np.sum(annual_mean_temperature_north_) / ntimesteps
    average_temperature_south_ = np.sum(annual_mean_temperature_south_) / ntimesteps
    average_temperature_total_ = np.sum(annual_mean_temperature_total_) / ntimesteps

    Functions.plot_annual_temperature_north_south(annual_mean_temperature_north_, annual_mean_temperature_south_,
                                        annual_mean_temperature_total_, average_temperature_north_,
                                        average_temperature_south_, average_temperature_total_, "Annual Ocean  Surface Temperature")
    
    
    annual_mean_temperature_north_land = [Functions.calc_mean_ocn_north(T_S_land[:, :, t], mesh.area) for t in range(ntimesteps)]
    annual_mean_temperature_south_land = [Functions.calc_mean_ocn_south(T_S_land[:, :, t], mesh.area) for t in range(ntimesteps)]
    annual_mean_temperature_total_land = [Functions.calc_mean_ocn(T_S_land[:, :, t], mesh.area) for t in range(ntimesteps)]

    average_temperature_north_land = np.sum(annual_mean_temperature_north_land) / ntimesteps
    average_temperature_south_land = np.sum(annual_mean_temperature_south_land) / ntimesteps
    average_temperature_total_land = np.sum(annual_mean_temperature_total_land) / ntimesteps

    Functions.plot_annual_temperature_north_south(annual_mean_temperature_north_land, annual_mean_temperature_south_land,
                                        annual_mean_temperature_total_land, average_temperature_north_land,
                                        average_temperature_south_land, average_temperature_total_land, "Annual Land Surface Temperature")


    annual_temperature_cologne = (Surface_Temp[14, 67, :] + Surface_Temp[14, 68, :]) / 2
    #annual_temperature_cologne = (Surface_Temp[20, 101, :] + Surface_Temp[21, 101, :]) / 2
    average_temperature_cologne = np.sum(annual_temperature_cologne) / ntimesteps

    Functions.plot_temperature_time(annual_temperature_cologne, average_temperature_cologne, "Annual temperature in Cologne")
    
    T_OCN_zero = copy.copy(T_OCN)
    T_OCN_zero[np.isnan(T_OCN_zero)] = 0
    
    T_S_Ocean_zero = copy.copy(T_S_Ocean)
    T_S_Ocean_zero[np.isnan(T_S_Ocean_zero)] = 0
    
    Functions.plot_inital_temperature(np.mean(T_OCN_zero, axis=2), mesh.Tf , "Ocean temperature")
    Functions.plot_inital_temperature(np.mean(Surface_Temp, axis=2), mesh.Tf , "Surface temperature")
    Functions.plot_inital_temperature(np.mean(T_ATM, axis=2), mesh.Tf , "Atmospheretemperature")
    
    Functions.plot_annual_temperature(np.mean(np.mean(T_OCN_zero, axis=1), axis=1),np.mean(T_OCN_zero),"Annual Temp Latitude Ocean ")
    Functions.plot_annual_temperature(np.mean(np.mean(T_S_Ocean_zero, axis=1), axis=1),np.mean(T_S_Ocean_zero),"Annual Temp Latitude Ocean Surface ")
    Functions.plot_annual_temperature(np.mean(np.mean(Surface_Temp, axis=1), axis=1),np.mean(Surface_Temp),"Annual Temp Latitude Surface ")

    #Plot for ice thickness (Average is only calculated where ice is present)
    H_I_nan = H_I
    H_I_nan[H_I_nan == 0] = np.NAN

    annual_mean_H_I_north =[Functions.calc_mean_ocn_north(H_I_nan[:,:,t], mesh.area) for t in range(ntimesteps)]
    Functions.plot_ice_thickness_time(annual_mean_H_I_north, np.mean(annual_mean_H_I_north) , "Mean Ice Thickness North")
    
    annual_mean_H_I_south =[Functions.calc_mean_ocn_south(H_I_nan[:,:,t], mesh.area) for t in range(ntimesteps)]
    Functions.plot_ice_thickness_time(annual_mean_H_I_south, np.mean(annual_mean_H_I_south) , "Mean Ice Thickness South") 
    
    
    
    Functions.plot_inital_temperature(Surface_Temp[:,:,1], mesh.Tf , "Surface temperature March")
     
    
    #Plot temperature for each time stepT
    filenames = []
    for ts in range(48):
        filename_ = Functions.plot_temperature(Surface_Temp, Geo_dat[:,:,ts], ts, show_plot=False)
        filenames.append(filename_)
    
    import imageio
    # Build GIF
    frames = [imageio.v3.imread(filename_) for filename_ in filenames]
    imageio.mimsave("annual_temperature.gif", frames)

    import os
    # Remove files
    for filename_ in set(filenames):
        os.remove(filename_)   
        
  