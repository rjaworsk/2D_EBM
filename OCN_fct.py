#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:11:54 2023

@author: ricij
"""


import numpy as np
from scipy import special

import Functions

#used for an EBM with only ocean --> now we use function in Functions
def calc_coalbedo(phi, phi_i_n, phi_i_s, mesh): #Calculation of the coalbedo where phi_i_n and phi_i_s are the latitudes wich have ice and are closest to the equator
    a0 = 0.72
    ai = 0.36
    a2 = (a0 - ai)/((np.pi/2)**2)
    equator = int((mesh.n_latitude - 1) / 2)
    coalbedo = np.zeros((mesh.n_latitude,mesh.n_longitude))
    
    for i in range(mesh.n_longitude):
        north_h =  0.5 * ((a0-a2*phi[equator:(len(phi))]**2 + ai) - (a0-a2*phi[equator:len(phi)]**2 - ai) * special.erf((phi[equator:len(phi)]-phi_i_n[i])/0.04))
    
        south_h = 0.5 * ((a0-a2*phi[0:equator]**2 + ai) - (a0-a2*phi[0:equator]**2 - ai) * special.erf((phi_i_s[i]-phi[0:equator])/0.04))
    
        coalbedo_concat = np.concatenate((south_h, north_h))
        
        coalbedo[:,i] = coalbedo_concat
    
    return coalbedo


#Ocean Circulation (flux from the depth of the ocean) --> only imrportant for actual ocean (not lakes)
#Analog to the function in the paper by Aylmer --> is similar for every longitude
def BasalFlux(phi, mesh):
    def f(phi):
        return -(1.3E16/(2*np.pi*6.37E6**2)) * np.cos(phi)**8 * (1-11*np.sin(phi)**2)
    def f_schlange(phi):
        return (1-3*np.cos(2*phi))/4
    F_bp = 2
    F_b = np.zeros((mesh.n_latitude, mesh.n_longitude))
    for i in range(mesh.n_longitude): 
        F_b[:,i] = f(phi) + F_bp * f_schlange(phi)        
    return F_b


#Jacobi Matrix for solving the Ocean-EBM 
def calc_jacobian_ocn(mesh, diffusion_coeff, heat_capacity, phi, Ocean_boundary):
    jacobian = np.zeros((mesh.ndof, mesh.ndof))
    test_temperature = np.zeros((mesh.n_latitude, mesh.n_longitude))

    index = 0
    for j in range(mesh.n_latitude):
        for i in range(mesh.n_longitude):
             
            if (Ocean_boundary[j,i] == 5) or (Ocean_boundary[j,i] == 2) or (Ocean_boundary[j,i] == 0): #only calculate diffusion for Ocean 
                test_temperature[j, i] = 1.0
            
                diffusion_op = Functions.calc_diffusion_operator_ocn(mesh, diffusion_coeff, test_temperature, Ocean_boundary)
           
                op = diffusion_op/heat_capacity
                   
            elif (Ocean_boundary[j,i] == 1) or (Ocean_boundary[j,i] == 3) : #for land and snow we do not have diffusion of the ocean
                op = np.zeros((mesh.n_latitude,mesh.n_longitude)) 
               
            # Convert matrix to vector
            jacobian[:, index] = op.flatten()

            # Reset test_temperature
            test_temperature[j, i] = 0
            index += 1
    
    return jacobian


def timestep_euler_backward_ocn(solve, delta_t, T_OCN, T_S, T_ATM, mesh, heat_capacity, solar_forcing, F_b, H_I, Geo_dat, Lakes):
    
    source_terms = ((solar_forcing - (mesh.A_up) - (mesh.B_up) * T_S + (mesh.A_dn) + (mesh.B_dn) * T_ATM + F_b) * 1/heat_capacity) * (H_I <=0)
    
    for i in range (mesh.n_latitude):
        for j in range(mesh.n_longitude):
            if Geo_dat[i,j] == 1 or Geo_dat[i,j] == 3:
                source_terms[i,j] = 0
            if  Lakes[i,j] == 4:  #because lakes to not have a deep ocean circulation --> no term Fb should be in the source terms for those points
                  F_b[i,j] = 0
    
    T_OCN_New = np.reshape(solve((T_OCN + delta_t * source_terms).flatten()), (mesh.n_latitude, mesh.n_longitude))

    return T_OCN_New 

   