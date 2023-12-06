#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:28:15 2023

@author: ricij
"""


import numpy as np

import Functions

#Jacobian Matrix for the atmosphere 
def calc_jacobian_atm(mesh, diffusion_coeff, heat_capacity, phi, Geo_dat, Aup, Bup, Adn, Bdn , Aolr, Bolr):
    jacobian = np.zeros((mesh.ndof, mesh.ndof))
    test_temperature = np.zeros((mesh.n_latitude, mesh.n_longitude))

    index = 0
    for j in range(mesh.n_latitude):
        for i in range(mesh.n_longitude):
            test_temperature[j, i] = 1.0
            diffusion_op = Functions.calc_diffusion_operator_atm(mesh, diffusion_coeff, test_temperature)
           
            if Geo_dat[j,i] == 1 or Geo_dat[j,i] == 3: #Adjustments for the different surface fluxes on land
                if Geo_dat[j,i] == 1:
                    op = (diffusion_op + (-(Bdn) - Bolr) * test_temperature) / heat_capacity
                   
                elif    Geo_dat[j,i] == 3:
                     
                    op = (diffusion_op + (-(Bdn) - Bolr) * test_temperature) / heat_capacity
               
            else: 
               op = (diffusion_op + (-mesh.B_dn-mesh.B_olr) * test_temperature) / heat_capacity 
               
               
            # Convert matrix to vector
            jacobian[:, index] = op.flatten()

            # Reset test_temperature
            test_temperature[j,i] = 0.0
            index += 1

    return jacobian



    
def timestep_euler_backward_atm(solve, delta_t,  T_ATM, Surface_Temp, mesh, heat_capacity, c02_warming, Geo_dat, Aup, Bup, Adn, Bdn,Aolr, Bolr): 
 
    source_terms = np.zeros((mesh.n_latitude, mesh.n_longitude))
    for i in range(mesh.n_latitude): 
        for j in range(mesh.n_longitude): 
            if Geo_dat[i,j] == 1 or Geo_dat[i,j] == 3: #Adjustments for different surface fluxes on land
                  if    Geo_dat[i,j] == 1:
                      
                       source_terms[i,j] = (Aup  - Adn - Aolr  + c02_warming  + Bup * Surface_Temp[i,j]) / heat_capacity[i,j] 
                   
                  elif Geo_dat[i,j] == 3:
                      source_terms[i,j] = (Aup  - Adn - Aolr  + c02_warming  + Bup * Surface_Temp[i,j]) / heat_capacity[i,j] 
              
                      if i > 32: 
                      
                         source_terms[i,j] = (Aup  - Adn-200 - Aolr  + c02_warming  + Bup * Surface_Temp[i,j]) / heat_capacity[i,j] 
                 
            else:
                source_terms[i,j] = (mesh.A_up - mesh.A_dn - mesh.A_olr + c02_warming  + mesh.B_up * Surface_Temp[i,j]) / heat_capacity[i,j] 
    
   
    T_ATM_New = np.reshape(solve((T_ATM + delta_t * source_terms).flatten()), (mesh.n_latitude, mesh.n_longitude))  
  
    return T_ATM_New
    
