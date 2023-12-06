#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:56:55 2023

@author: ricij
"""
import numpy as np
import Functions

#Jacobian Matrix for the land 
# def calc_jacobian_s(mesh, diffusion_coeff, heat_capacity, phi, Surface_boundary):
#     jacobian = np.zeros((mesh.ndof, mesh.ndof))
#     test_temperature = np.zeros((mesh.n_latitude, mesh.n_longitude))

#     index = 0
#     for j in range(mesh.n_latitude):
#         for i in range(mesh.n_longitude):
#             if (Surface_boundary[j,i] == 1) or (Surface_boundary[j,i] == 3) or (Surface_boundary[j,i] == 0): #Only calculate diffusion for land 
#               test_temperature[j, i] = 1.0

#               diffusion_op = Functions.calc_diffusion_operator_land(mesh, diffusion_coeff, test_temperature, Surface_boundary)
               
#               #op = (diffusion_op - 11.07 * test_temperature)/mesh.C_s
            
#               #op = (diffusion_op - mesh.B_up * test_temperature)/mesh.C_s # Original
              
#               op = (diffusion_op - 2.15 * test_temperature)/mesh.C_s # wie in VL
                
#             else: # Do not have diffusion for the ocean or sea ice here
                
#               op = np.zeros((mesh.n_latitude, mesh.n_longitude))
              
#             # Convert matrix to vector
#             jacobian[:, index] = op.flatten()

#             # Reset test_temperature
#             test_temperature[j, i] = 0.0
#             index += 1

#     return jacobian

def calc_jacobian_s(mesh, diffusion_coeff, heat_capacity, phi, Surface_boundary):
    jacobian = np.zeros((2964, 2964))
    #jacobian = np.zeros((6634, 6634))
    #jacobian = np.zeros((11759, 11759))
    #jacobian = np.zeros((18270, 18270))
    test_temperature = np.zeros((mesh.n_latitude, mesh.n_longitude))
    Geo_vector = Surface_boundary.flatten()

    index = 0
    for j in range(mesh.n_latitude):
        for i in range(mesh.n_longitude):
            if (Surface_boundary[j,i] == 1) or (Surface_boundary[j,i] == 3) or (Surface_boundary[j,i] == 0): #Only calculate diffusion for land 
              test_temperature[j, i] = 1.0

              diffusion_op = Functions.calc_diffusion_operator_land(mesh, diffusion_coeff, test_temperature, Surface_boundary)
               
              #op = (diffusion_op - 7.07 * test_temperature)/mesh.C_s
            
              op = (diffusion_op - mesh.B_up * test_temperature)/mesh.C_s # Original
              
             # op = (diffusion_op - 2.15 * test_temperature)/mesh.C_s # wie in VL
                
            
                
              op_flat = op.flatten()
              
              # Convert matrix to vector
             
              l = []
              for k in range(mesh.ndof):
                  if Geo_vector[k] == 2 or Geo_vector[k] == 5:
                      l.append(k)
                      
              for ind in sorted(l, reverse=True):
                  op_flat = np.delete(op_flat,ind)
              jacobian[:, index] = op_flat


              # Reset test_temperature
              test_temperature[j, i] = 0.0
              index += 1

    return jacobian


# def timestep_euler_backward_s(solve, delta_t, T_S, T_ATM, t, mesh, solar_forcing, Geo_dat, heat_capacity):
    
#     # source_terms = ((solar_forcing - 350.305 + 269.739 + 4.009 * T_ATM) / mesh.C_s) 
#     # source_terms = ((solar_forcing - mesh.A_up + mesh.A_dn + mesh.B_dn * T_ATM) / mesh.C_s) # Original
#     source_terms = (solar_forcing -210.3)/mesh.C_s #VL
#     for i in range (mesh.n_latitude):
#         for j in range(mesh.n_longitude):
#             if Geo_dat[i,j] == 2 or Geo_dat[i,j] == 5:
#                 source_terms[i,j] = 0            
#     T_S_New = np.reshape(solve((T_S + delta_t * source_terms).flatten()), (mesh.n_latitude, mesh.n_longitude))
#     return T_S_New 

def timestep_euler_backward_s(solve, delta_t, T_S, T_ATM, t, mesh, solar_forcing, Geo_dat, heat_capacity):
    
    Geo_vector = Geo_dat.flatten()
   # source_terms = ((solar_forcing - 350.305 + 269.739 + 4.009 * T_ATM) / mesh.C_s) 
    #source_terms = ((solar_forcing - mesh.A_up + mesh.A_dn + mesh.B_dn * T_ATM) / mesh.C_s) # Original
   # source_terms = (solar_forcing -210.3)/mesh.C_s #VL
    source_terms = (solar_forcing - mesh.A_up + mesh.A_dn + mesh.B_dn * T_ATM) / mesh.C_s
    for i in range (mesh.n_latitude):
        for j in range(mesh.n_longitude):
            if Geo_dat[i,j] == 2 or Geo_dat[i,j] == 5:
                source_terms[i,j] = 0     
                
    b  = (T_S + delta_t * source_terms).flatten()           
    l = []
    for k in range(mesh.ndof):
          if Geo_vector[k] == 2 or Geo_vector[k] == 5:
              l.append(k)
             
    for ind in sorted(l, reverse=True):
          b = np.delete(b,ind)
            
    T_S_New = solve(b)
    
    z = np.zeros(mesh.ndof)
    t  = 0
    o = 0
    for i in range(mesh.ndof):
        if i < 7606:
      #  if i < 17169:  
       # if i < 30571:    
        #if i < 47494:        
            if l[t] <= i:
                z[i] = 0
                t= t+1
               
            else:
                z[i] = T_S_New[o]
                o = o+1
                
        else: 
            z[i] = T_S_New[o]
            o = o+1

    Land_Surface =  np.reshape(z, (mesh.n_latitude, mesh.n_longitude))
    return  Land_Surface

