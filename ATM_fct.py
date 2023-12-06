#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:10:34 2023

@author: ricij
"""


import numpy as np

import Functions

#Jacobian Matrix for the atmosphere 
def calc_jacobian_atm(mesh, diffusion_coeff, heat_capacity, phi, Geo_dat):
    jacobian = np.zeros((mesh.ndof, mesh.ndof))
    test_temperature = np.zeros((mesh.n_latitude, mesh.n_longitude))

    index = 0
    for j in range(mesh.n_latitude):
        for i in range(mesh.n_longitude):
            test_temperature[j, i] = 1.0
            diffusion_op = Functions.calc_diffusion_operator_atm(mesh, diffusion_coeff, test_temperature)
           
            if Geo_dat[j,i] == 1 or Geo_dat[j,i] == 3: #Adjustments for the different surface fluxes on land
                op = (diffusion_op + (-mesh.B_dn-mesh.B_olr) * test_temperature) / heat_capacity  
               # op = (diffusion_op + (-6.04922-mesh.B_olr) * test_temperature) / heat_capacity  # Tuning mit Least Squares
               # op = (diffusion_op + (-2.62776805e-14-mesh.B_olr) * test_temperature) / heat_capacity
            else: 
                op = (diffusion_op + (- mesh.B_dn-mesh.B_olr) * test_temperature) / heat_capacity  
                
            # Convert matrix to vector
            jacobian[:, index] = op.flatten()

            # Reset test_temperature
            test_temperature[j,i] = 0.0
            index += 1

    return jacobian



    
def timestep_euler_backward_atm(solve, delta_t,  T_ATM, T_S, mesh, heat_capacity, c02_warming, Geo_dat): 
 
    source_terms = np.zeros((mesh.n_latitude, mesh.n_longitude))
    for i in range(mesh.n_latitude): 
        for j in range(mesh.n_longitude): 
            if Geo_dat[i,j] == 1 or Geo_dat[i,j] == 3: #Adjustments for different surface fluxes on land
                #source_terms[i,j] = (  489.543109  -   328.30824  - mesh.A_olr + c02_warming  - 6.71741559  * T_S[i,j]) / heat_capacity[i,j] 
                source_terms[i,j] = (mesh.A_up - mesh.A_dn - mesh.A_olr + c02_warming  +  mesh.B_up * T_S[i,j]) / heat_capacity[i,j] 
               # source_terms[i,j] = (386.92082 - 334.6069 - mesh.A_olr + c02_warming  +10.6018 * T_S[i,j]) / heat_capacity[i,j] 
            else:
                source_terms[i,j] = (mesh.A_up - mesh.A_dn - mesh.A_olr + c02_warming  + mesh.B_up * T_S[i,j]) / heat_capacity[i,j] 
    
    T_ATM_New = np.reshape(solve((T_ATM + delta_t * source_terms).flatten()), (mesh.n_latitude, mesh.n_longitude))  
  
    return T_ATM_New
    
