#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:10:34 2023

@author: ricij
"""


import numpy as np

import Functions



def calc_jacobian_atm(mesh, diffusion_coeff, heat_capacity, phi):
    jacobian = np.zeros((mesh.ndof, mesh.ndof))
    test_temperature = np.zeros((mesh.n_latitude, mesh.n_longitude))

    index = 0
    for j in range(mesh.n_latitude):
        for i in range(mesh.n_longitude):
            test_temperature[j, i] = 1.0
            diffusion_op = Functions.calc_diffusion_operator_atm(mesh, diffusion_coeff, test_temperature)
           
            op = (diffusion_op + (-mesh.B_dn-mesh.B_olr) * test_temperature) / heat_capacity
            
            # Convert matrix to vector
            jacobian[:, index] = op.flatten()

            # Reset test_temperature
            test_temperature[j,i] = 0.0
            index += 1

    return jacobian



    
def timestep_euler_backward_atm(solve, delta_t,  T_ATM, T_S, t,  mesh, heat_capacity, c02_warming): 

    source_terms = (mesh.A_up - mesh.A_dn - mesh.A_olr + c02_warming  + mesh.B_up * T_S) / heat_capacity 
    
    T_ATM_New = np.reshape(solve((T_ATM + delta_t * source_terms).flatten()), (mesh.n_latitude, mesh.n_longitude))  
  
    return T_ATM_New
    
