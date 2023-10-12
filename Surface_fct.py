#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:56:55 2023

@author: ricij
"""
import numpy as np
import Functions


def calc_jacobian_s(mesh, diffusion_coeff, heat_capacity, phi, Surface_boundary):
    jacobian = np.zeros((mesh.ndof, mesh.ndof))
    test_temperature = np.zeros((mesh.n_latitude, mesh.n_longitude))

    index = 0
    for j in range(mesh.n_latitude):
        for i in range(mesh.n_longitude):
            if (Surface_boundary[j,i] == 1) or (Surface_boundary[j,i] == 3) or (Surface_boundary[j,i] == 0): #only calculate diffusion for Ocean 
                test_temperature[j, i] = 1.0
               
                diffusion_op = Functions.calc_diffusion_operator_land(mesh, diffusion_coeff, test_temperature, Surface_boundary)
                
                op = (diffusion_op - mesh.B_up * test_temperature)/heat_capacity
                
            else: 
                
                op = np.zeros((mesh.n_latitude, mesh.n_longitude))
            # Convert matrix to vector
            jacobian[:, index] = op.flatten()

            # Reset test_temperature
            test_temperature[j, i] = 0.0
            index += 1

    return jacobian


def timestep_euler_backward_s(solve, delta_t, T_S, T_ATM, t, mesh, solar_forcing, Geo_dat, heat_capacity):
    
    source_terms = ((solar_forcing - mesh.A_up + mesh.A_dn + mesh.B_dn * T_ATM  ) / heat_capacity)  
   
    for i in range (mesh.n_latitude):
        for j in range(mesh.n_longitude):
            if (Geo_dat[i,j]) ==5 or Geo_dat[i,j] == 2:
                source_terms[i,j] = 0 
                
    T_S_New = np.reshape(solve((T_S + delta_t * source_terms).flatten()), (mesh.n_latitude, mesh.n_longitude))
    return T_S_New 

