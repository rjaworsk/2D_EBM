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
import math
import scipy

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
  
def ice_edge(H_I, phi, Geo_dat):    
    nlatitude, nlongitude = H_I.shape
    ice_edge_latitude_n = np.zeros(nlongitude)
    ice_edge_latitude_s = np.zeros(nlongitude)
    lat_index_n = np.zeros(nlongitude)
    lat_index_s = np.zeros(nlongitude)
    
    for i in range(nlongitude): #calculation of ice edge latitude for each longitude
        
        if H_I[0,i] == 0 and (Geo_dat[0,i] == 5 or Geo_dat[0,i] == 2 ): #calc ice edge north 
            index_n = 0
            ice_latitude_n = phi[0]
            
        elif H_I[0,i] != 0 or (Geo_dat[0,i] == 3 or Geo_dat[0,i] == 1) :
            index_n = 1
            
            while (H_I[index_n,i] > 0 or (Geo_dat[index_n,i] == 3 or Geo_dat[index_n,i] == 1)): 
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
            
            while (H_I[index_s,i] > 0 or (Geo_dat[index_s,i] == 3 or Geo_dat[index_s,i] == 1)): 
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
  

def surface_temp_land(T_ATM, solar_forcing, phi, mesh, Geo_dat, T_S_land, delta_t): #calc surface temp of land
    T_S_land_new = np.zeros((mesh.n_latitude, mesh.n_longitude))
   
    RHS = 1/mesh.C_s * (solar_forcing + mesh.A_dn + mesh.B_dn * T_ATM - mesh.A_up - mesh.B_up * T_S_land )
   # RHS = 1/mesh.C_s * (solar_forcing + mesh.B_dn * T_ATM -110.3 - mesh.B_up * T_S_land )
    T_S_land_new = T_S_land + delta_t * RHS
          
    return T_S_land_new

def surface_temp(T_ATM, T_OCN, H_I, solar_forcing, phi, mesh, Geo_dat): #calculate surface temp of ocean
    T_S = copy.copy(T_OCN)
    
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
    
    H_I_new = H_I - delta_t * (1/mesh.Lf * (-mesh.A_up - mesh.B_up * T_S + mesh.A_dn + mesh.B_dn * T_ATM + Fb + solar_forcing) * (H_I >0))
    
    return H_I_new



def compute_equilibrium(mesh, diffusion_coeff_atm, heat_capacity_atm, T_ATM_0, T_OCN_0, T_S_0, P_ocn,
                          diffusion_coeff_ocn, heat_capacity_ocn, solar_forcing, phi, true_longitude,n_timesteps, Geo_dat, heat_capacity_s, diffusion_coeff_s, Ocean_boundary, Lakes, Surface_boundary,  c02_warming, max_iterations=100, rel_error=1e-5, verbose=True):
    # Step size
    delta_t = 1 / ntimesteps
    
    #Inital Conditions
    T_ATM = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    #T_ATM[:,:,-1] = T_ATM_0 #inital conditions from paper
    T_OCN = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
   # T_OCN[:,:,-1] = T_OCN_0 #inital conditions from paper
    T_S = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    #T_S[:,:,-1] = T_S_0 #inital conditions from paper
    T_S_land = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
    #T_S_land[:,:,-1] = T_S_0
    H_I = np.zeros((mesh.n_latitude, mesh.n_longitude, ntimesteps))
   # H_I[:,:,-1] = H_I_0 #inital conditions from paper
   
    Surface_Temp = np.zeros((mesh.n_latitude,mesh.n_longitude, ntimesteps))
   
    
    # Area-mean in every time step
    temp_atm = np.zeros(ntimesteps)
    temp_ocn=  np.zeros(ntimesteps)
    temp_s =  np.zeros(ntimesteps)
    temp_H_I =  np.zeros(ntimesteps)
    
    phi_index_s = np.zeros((mesh.n_longitude,ntimesteps))
    phi_index_n = np.zeros((mesh.n_longitude,ntimesteps))
    

    # Average temperature over all time steps from the previous iteration to approximate the error
    old_avg_atm = 0
    old_avg_ocn = 0
    old_avg_s = 0
    old_avg_H_I = 0
    
    
    Fb = OCN_fct.BasalFlux(phi, mesh) #additional ocean flux
    Hml = P_ocn.Hml_const * np.ones((mesh.n_latitude,mesh.n_longitude))
    
    # Construct and factorize Jacobian for the atmosphere
    jacobian_atm = ATM_fct.calc_jacobian_atm(mesh, diffusion_coeff_atm, P_atm.heat_capacity, phi)
    m, n = jacobian_atm.shape
    eye = sparse.eye(m, n, format="csc")
    jacobian_atm = sparse.csc_matrix(jacobian_atm)
    jacobian_atm = sparse.linalg.factorized(eye - delta_t * jacobian_atm)   
    
    # Construct and factorize Jacobian for the ocean
    jacobian_ocn = OCN_fct.calc_jacobian_ocn(mesh, diffusion_coeff_ocn, heat_capacity_ocn, phi, Ocean_boundary)
    m, n = jacobian_ocn.shape
    eye = sparse.eye(m, n, format="csc")
    jacobian_ocn = sparse.csc_matrix(jacobian_ocn)
    jacobian_ocn = sparse.linalg.factorized(eye - delta_t * jacobian_ocn)
    
   
    # Construct and factorize Jacobian for the land
    jacobian_s = Surface_fct.calc_jacobian_s(mesh, diffusion_coeff_s, heat_capacity_s, phi, Surface_boundary)
    m, n = jacobian_s.shape
    eye = sparse.eye(m, n, format="csc")
    jacobian_s = sparse.csc_matrix(jacobian_s)
    jacobian_s = sparse.linalg.factorized(eye - delta_t * jacobian_s)
    
    # Compute insolation
    insolation = Functions.calc_insolation(phi, true_longitude)
    
    
    timer = Timer()
    for i in range(max_iterations):
        print("Iteration: ",i)
        timer.start()
        for t in range(ntimesteps):    
            
            phi_index_s[:,t], phi_i_s, phi_index_n[:,t], phi_i_n = ice_edge(H_I[:,:,t-1], phi, Geo_dat)  # new Ice_Edge Index 
            
            Geo_dat = Functions.change_geo_dat(Geo_dat, H_I[:,:,t-1], mesh) #calc new ice distribution
            
            Ocean_boundary = Functions.Get_Ocean_Boundary_Distribution(Geo_dat) #get new ocean boundary for diffusion process
            
            coalbedo = Functions.calc_coalbedo(Geo_dat, phi_i_n, phi_i_s) #new coalbdeo dependant on the ice_edge 
            
            solar_forcing  = Functions.calc_solar_forcing(insolation[:,t-1],coalbedo, mesh)        
            
            T_ATM[:,:,t] = ATM_fct.timestep_euler_backward_atm(jacobian_atm, 1/ntimesteps, T_ATM[:,:,t-1], Surface_Temp[:,:,t-1], t, mesh, P_atm.heat_capacity, c02_warming)
            
            T_OCN[:,:,t] = OCN_fct.timestep_euler_backward_ocn(jacobian_ocn, 1 / ntimesteps, T_OCN[:,:,t-1], T_S[:,:,t-1], T_ATM[:,:,t-1], t, mesh, heat_capacity_ocn, solar_forcing, Fb, H_I[:,:,t-1], Geo_dat, Ocean_boundary, Lakes)
            
            H_I[:,:,t] = timestep_euler_forward_H_I(mesh,T_S[:,:,t-1], T_ATM[:,:,t-1], Fb, solar_forcing, H_I[:,:,t-1], t, delta_t)
            
            T_OCN[:,:,t], H_I[:,:,t] = FreezeAndMelt(T_OCN[:,:,t], H_I[:,:,t], Hml, mesh)
            
            solar_forcing_new = Functions.calc_solar_forcing(insolation[:,t],coalbedo, mesh)   
            
            T_S_land[:,:,t] = Surface_fct.timestep_euler_backward_s(jacobian_s, delta_t, T_S_land[:,:,t-1], T_ATM[:,:,t], t, mesh, solar_forcing_new, Geo_dat,  heat_capacity_s)
    
            #T_S_land[:,:,t] = surface_temp_land(T_ATM[:,:,t], solar_forcing_new, phi, mesh, Geo_dat, T_S_land[:,:,t-1], delta_t)
            
            T_S[:,:,t] = surface_temp(T_ATM[:,:,t], T_OCN[:,:,t], H_I[:,:,t], solar_forcing_new, phi, mesh, Geo_dat) #original surface temp from Ocean Model --> is just for the ocean surface temperatue
           
            Surface_Temp[:,:,t] = Functions.unite_surface_temp(T_S[:,:,t] , T_S_land[:,:,t] , mesh, Geo_dat)
            
            temp_atm[t] = Functions.calc_mean(T_ATM[:,:,t], mesh.area)
            temp_ocn[t] = Functions.calc_mean_ocn(T_OCN[:,:,t], mesh.area)
            temp_s[t] = Functions.calc_mean(Surface_Temp[:,:,t], mesh.area)
            temp_H_I[t] = Functions.calc_mean_ocn(H_I[:,:,t], mesh.area)
            
        
        timer.stop("one year")
        avg_temperature_atm = np.sum(temp_atm) / ntimesteps
        avg_temperature_ocn = np.sum(temp_ocn) / ntimesteps
        avg_temperature_s = np.sum(temp_s) / ntimesteps
        avg_H_I = np.sum(temp_H_I) / ntimesteps
      
        print("Fehler Eisdicke: ",np.abs(avg_H_I - old_avg_H_I))
        print("Fehler ATMO: ",np.abs(avg_temperature_atm - old_avg_atm))
        print("Fehler Surface: ",np.abs(avg_temperature_s - old_avg_s))
        print("Fehler Ocean: ",np.abs(avg_temperature_ocn - old_avg_ocn))
        if (np.abs(avg_temperature_atm - old_avg_atm)< rel_error) and (np.abs(avg_temperature_ocn - old_avg_ocn)< rel_error)  and  (np.abs(avg_temperature_s - old_avg_s)< rel_error) and (np.abs(avg_H_I - old_avg_H_I) < rel_error):
            # We can assume that the error is sufficiently small now.
            verbose and print("Equilibrium reached!")
            
            break
        
        else:
              old_avg_atm = avg_temperature_atm
              old_avg_ocn = avg_temperature_ocn
              old_avg_s = avg_temperature_s
              old_avg_H_I = avg_H_I
  
         
    return  T_ATM, T_S_land, T_S, T_OCN, H_I, Surface_Temp, phi_index_s, phi_index_n
       
# Run code
if __name__ == '__main__':
    start = time.time()  
    initial = True
    file_path = '/Users/ricij/Documents/Universität/Master/Masterarbeit/VL_Klimamodellierung/input/The_World128x65.dat.txt'  
    Geo_dat = Functions.read_geography(file_path)
    Ocean_boundary = Functions.Get_Ocean_Boundary_Distribution(Geo_dat)
    Surface_boundary = Functions.Get_Surface_Boundary_Distribution(Geo_dat)

    mesh = Mesh()
     
    heat_capacity_s = Functions.calc_heat_capacity(Geo_dat)
   

    diffusion_coeff_s = Functions.calc_diffusion_coefficients(Geo_dat) #noch nicht eingebaut
    
    Lakes =  Functions.LandDstr_wLakes(Geo_dat)
    
    ntimesteps = 48 
    dt = 1/ ntimesteps
    ecc= 0.016740 #old ecc
    true_longitude = Functions.calc_lambda(dt,  ntimesteps, ecc =  0.016740, per = 1.783037)
    
   
    co2_ppm = 315.0
    c02_warming = Functions.calc_radiative_cooling_co2(co2_ppm)
  
    
    phi = np.linspace(np.pi/2,-np.pi/2,mesh.n_latitude) #from 90° south to 90° north

    phi_i_deg_n = 75 #inital value for the latitude of the ice-edge 
    phi_i_deg_s = 75 
    
    P_atm = P_atm() #Parameters for the atmosphere
    P_ocn = P_ocn() #Parameters for the ocean
    
    diffusion_coeff_atm = P_atm.heat_capacity * P_atm.diffusion_coeff /mesh.RE**2 #Diffusion coefficient
    
    heat_capacity_ocn = P_ocn.c_O * P_ocn.rhoo * P_ocn.Hml_const * np.ones((mesh.n_latitude, mesh.n_longitude))  # Hml can also be variable
    diffusion_coeff_ocn = heat_capacity_ocn * P_ocn.K_O / mesh.RE**2  #Diffusion coefficient
   
    
    #Inital Conditions
    if initial == True: 
        T_ATM_0 = np.ones((mesh.n_latitude,mesh.n_longitude))
        T_OCN_0 = np.ones((mesh.n_latitude,mesh.n_longitude))
        H_I_0 = np.ones((mesh.n_latitude,mesh.n_longitude))
        
        B = 3/((np.pi/2) - phi_i_deg_n * np.pi/180)
        A = -B *phi_i_deg_n * np.pi/180
        
        for i in range(mesh.n_longitude):
           T_ATM_0[:,i] = 0.5 * (-15 + 35 * np.cos(2*phi))
           T_OCN_0[:,i] = 0.5 * (28.2 + 31.8 * np.cos(180*phi/phi_i_deg_n))
           H_I_0[:,i] = A + B * phi
       
    
        H_I_0 = H_I_0 * (H_I_0 > 0) #because ice thickness cannot be negative
        H_I_0 = H_I_0[::-1]  + H_I_0
        T_OCN_0 = T_OCN_0 * (H_I_0 <= 0) + mesh.Tf * (H_I_0 > 0)
        
        Functions.plot_inital_temperature(T_OCN_0, mesh.Tf , "Ocean inital temperature") 
        Functions.plot_inital_temperature(T_ATM_0, mesh.Tf , "Atmosphere inital temperature")
        
        
        phi_index_s, phi_i_s, phi_index_n, phi_i_n = ice_edge(H_I_0, phi, Geo_dat)
        coalbedo_ocn = Functions.calc_coalbedo(Geo_dat, phi_i_n, phi_i_s)
        solar_forcing = np.zeros((mesh.n_latitude,mesh.n_longitude))
        insolation = Functions.calc_insolation(phi, true_longitude)
        for i in range(mesh.n_longitude):
            solar_forcing[:,i]  = insolation[:,0] * coalbedo_ocn[:,i]   
        T_S_0 = surface_temp(T_ATM_0, T_OCN_0, H_I_0, solar_forcing, phi, mesh, Geo_dat) #og
            
        Functions.plot_inital_temperature(T_S_0, mesh.Tf , "Surface inital temperature")
        
        
        for i in range (mesh.n_latitude):
            for j in range(mesh.n_longitude):
                if (Geo_dat[i,j]) ==1 or Geo_dat[i,j] == 3:
                    T_OCN_0[i,j] = 0 #None
                    
    ###### run model ######
    T_ATM, T_S_land, T_S, T_OCN, H_I, Surface_Temp, phi_index_s, phi_index_n  = compute_equilibrium( mesh, diffusion_coeff_atm, P_atm.heat_capacity, T_ATM_0, T_OCN_0, T_S_0, P_ocn, diffusion_coeff_ocn, heat_capacity_ocn, solar_forcing, phi, true_longitude, ntimesteps, Geo_dat, heat_capacity_s, diffusion_coeff_s, Ocean_boundary, Lakes, Surface_boundary,  c02_warming)

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


    annual_temperature_cologne = (Surface_Temp[14, 67, :] + Surface_Temp[14, 68, :]) / 2
    average_temperature_cologne = np.sum(annual_temperature_cologne) / ntimesteps

    Functions.plot_temperature_time(annual_temperature_cologne, average_temperature_cologne, "Annual temperature in Cologne")
    
    T_OCN_zero = copy.copy(T_OCN)
    T_OCN_zero[np.isnan(T_OCN_zero)] = 0
    Functions.plot_inital_temperature(np.mean(T_OCN_zero, axis=2), mesh.Tf , "Ocean temperature")
    Functions.plot_inital_temperature(np.mean(Surface_Temp, axis=2), mesh.Tf , "Surface temperature")
    Functions.plot_inital_temperature(np.mean(T_ATM, axis=2), mesh.Tf , "Atmospheretemperature")
    
    Functions.plot_annual_temperature(np.mean(np.mean(T_OCN_zero, axis=1), axis=1),np.mean(T_OCN_zero),"Annual Temp Latitude Ocean ")
    Functions.plot_annual_temperature(np.mean(np.mean(Surface_Temp, axis=1), axis=1),np.mean(Surface_Temp),"Annual Temp Latitude Surface ")

    #Plot for ice thickness (Average is only calculated where ice is present)
    H_I_nan = H_I
    H_I_nan[H_I_nan == 0] = np.NAN

    annual_mean_H_I_north =[Functions.calc_mean_ocn_north(H_I_nan[:,:,t], mesh.area) for t in range(ntimesteps)]
    Functions.plot_ice_thickness_time(annual_mean_H_I_north, np.mean(annual_mean_H_I_north) , "Mean Ice Thickness North")
    
    annual_mean_H_I_south =[Functions.calc_mean_ocn_south(H_I_nan[:,:,t], mesh.area) for t in range(ntimesteps)]
    Functions.plot_ice_thickness_time(annual_mean_H_I_south, np.mean(annual_mean_H_I_south) , "Mean Ice Thickness South") 