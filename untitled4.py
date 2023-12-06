#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:03:34 2023

@author: ricij
"""

@njit
def calc_diffusion_operator_inner_land(h, area, n_latitude, n_longitude, csc2, cot, diffusion_coeff, temperature):
    result = np.zeros(diffusion_coeff.shape)

    # North Pole
    factor = np.sin(h / 2) / (4 * np.pi * area[0])
    result[0, :] = factor * 0.5 * np.dot(diffusion_coeff[0, :] + diffusion_coeff[1, :],
                                          temperature[1, :] - temperature[0, :])

    # South Pole
    factor = np.sin(h / 2) / (4 * np.pi * area[-1])
    result[-1, :] = factor * 0.5 * np.dot(diffusion_coeff[-1, :] + diffusion_coeff[-2, :],
                                          temperature[-2, :] - temperature[-1, :])

    for i in range(n_longitude):
        # Only loop over inner nodes
        for j in range(1, n_latitude - 1):
            # There are the special cases of i=0 and i=n_longitude-1.
            # We have a periodic boundary condition, so for i=0, we want i-1 to be the last entry.
            # This happens automatically in Python when i=-1.
            # For i=n_longitude-1, we want i+1 to be 0.
            # For this, we define a variable ip1 (i plus 1) to avoid duplicating code.
            if i == n_longitude - 1:
                ip1 = 0
            else:
                ip1 = i + 1

            # Note that csc2 does not contain the values at the poles
            factor1 = csc2[j - 1] / h ** 2
            term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] +
                                (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *
                                temperature[j, i - 1] +
                                (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *
                                temperature[j, ip1])

            factor2 = 1 / h ** 2
            term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] +
                                (diffusion_coeff[j, i] - 0.25 *
                                (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) *
                                temperature[j - 1, i]
                                + (diffusion_coeff[j, i] + 0.25 *
                                  (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) *
                                temperature[j + 1, i])

            term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])

            result[j, i] = term1 + term2 + term3

    return result


def calc_diffusion_operator_land(mesh, diffusion_coeff, temperature, Surface_boundary):
    h = mesh.h
    area = mesh.area
    n_latitude = mesh.n_latitude
    n_longitude = mesh.n_longitude
    csc2 = mesh.csc2
    cot = mesh.cot

    return calc_diffusion_operator_inner_land(h, area, n_latitude, n_longitude, csc2, cot, diffusion_coeff, temperature, Surface_boundary )

@njit    
def calc_diffusion_operator_inner_land(h, area, n_latitude, n_longitude, csc2, cot, diffusion_coeff, temperature, Surface_boundary):
    result = np.zeros(diffusion_coeff.shape)

    # North Pole
    factor = np.sin(h / 2) / (4 * np.pi * area[0])
    #result[0, :] = factor * 0.5 * np.dot(diffusion_coeff[0, :] + diffusion_coeff[1, :], temperature[1, :] - temperature[0, :])
    result[0, :] = np.ones(128)  * np.nan
    # South Pole
    factor = np.sin(h / 2) / (4 * np.pi * area[-1])
    result[-1, :] = factor * 0.5 * np.dot(diffusion_coeff[-1, :] + diffusion_coeff[-2, :], temperature[-2, :] - temperature[-1, :])
   
    for i in range(n_longitude):
        # Only loop over inner nodes
        for j in range(1, n_latitude - 1):
            # There are the special cases of i=0 and i=n_longitude-1.
            # We have a periodic boundary condition, so for i=0, we want i-1 to be the last entry.
            # This happens automatically in Python when i=-1.
            # For i=n_longitude-1, we want i+1 to be 0.
            # For this, we define a variable ip1 (i plus 1) to avoid duplicating code.
            if i == n_longitude - 1:
                ip1 = 0
            else:
                ip1 = i + 1
            
            if Surface_boundary[j,i] == 0 :  
                term1 = 0
                term2 = 0
                term3 = 0
                if Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2: 
                    factor1 = csc2[j - 1] / h ** 2
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] +
                                       (diffusion_coeff[j, i] - 0.25 *
                                        (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) *
                                       temperature[j - 1, i]
                                       + (diffusion_coeff[j, i] + 0.25 *
                                          (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) *
                                       temperature[j, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j, i] - temperature[j - 1, i])
                if  Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2: 
                    factor2 = 1 / h ** 2
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] +
                                       (diffusion_coeff[j, i] - 0.25 *
                                        (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) *
                                       temperature[j, i]
                                       + (diffusion_coeff[j, i] + 0.25 *
                                          (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) *
                                       temperature[j + 1, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j, i])
                    
                if  Surface_boundary[j,i+1] == 5 or Surface_boundary[j,i+1] == 2: 
                    factor1 = csc2[j - 1] / h ** 2
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] +
                                       (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *
                                       temperature[j, i - 1] +
                                       (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *
                                       temperature[j, i])
                    
                if  Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2: 
                    factor1 = csc2[j - 1] / h ** 2
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] +
                                       (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *
                                       temperature[j, i] +
                                       (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *
                                       temperature[j, ip1])
                    
                if   (Surface_boundary[j+1,i] == 1 or  Surface_boundary[j+1,i] == 3) and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 3): 
                    factor2 = 1 / h ** 2
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] +
                                       (diffusion_coeff[j, i] - 0.25 *
                                        (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) *
                                       temperature[j - 1, i]
                                       + (diffusion_coeff[j, i] + 0.25 *
                                          (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) *
                                       temperature[j + 1, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])
                    
                if   (Surface_boundary[j,i+1] == 1 or  Surface_boundary[j,i+1] == 3) and (Surface_boundary[j,i-1] == 1 or Surface_boundary[j,i-1] == 3): 
                    factor1 = csc2[j - 1] / h ** 2
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] +
                                       (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *
                                       temperature[j, i - 1] +
                                       (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *
                                       temperature[j, ip1])
               
            elif (Surface_boundary[j,i] == 2 or Surface_boundary[j,i] == 5):   
                term1 = np.nan
                term2 = np.nan
                term3 = np.nan
                
            else:    
                # Note that csc2 does not contain the values at the poles
                factor1 = csc2[j - 1] /(h ** 2)
                term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1]))  * temperature[j, ip1])
    
                factor2 = 1 / (h ** 2)
                term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i]))  * temperature[j + 1, i])
    
                term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])

            result[j, i] = term1 + term2 + term3
            

    return result

##### OCEAN 

@njit    
def calc_diffusion_operator_inner_ocn(h, area, n_latitude, n_longitude, csc2, cot, diffusion_coeff, temperature, Ocean_boundary, k, l):
    result = np.zeros(diffusion_coeff.shape)

    # North Pole --> Boundary Condition
    factor = np.sin(h / 2) / (4 * np.pi * area[0])
    result[0, :] = factor * 0.5 * np.dot(diffusion_coeff[0, :] + diffusion_coeff[1, :], temperature[1, :] - temperature[0, :])

    # South Pole
    result[-1, :] = np.ones(128) * np.nan #No ocean at the south pole --> no diffusion
    for i in range(n_longitude):
        # Only loop over inner nodes
        for j in range(1, n_latitude - 1):
            # There are the special cases of i=0 and i=n_longitude-1.
            # We have a periodic boundary condition, so for i=0, we want i-1 to be the last entry.
            # This happens automatically in Python when i=-1.
            # For i=n_longitude-1, we want i+1 to be 0.
            # For this, we define a variable ip1 (i plus 1) to avoid duplicating code.
            if i == n_longitude - 1:
                ip1 = 0
            else:
                ip1 = i + 1
            
            if Ocean_boundary[j,i] == 0 :  
                if (Ocean_boundary[j,ip1] == 1 or  Ocean_boundary[j,ip1] == 3) and (Ocean_boundary[j,i-1] == 1 or Ocean_boundary[j,i-1] == 3) and (Ocean_boundary[j+1,i] == 1 or Ocean_boundary[j+1,i] == 3) and (Ocean_boundary[j-1,i] == 1 or Ocean_boundary[j-1,i] == 3):
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])
                    
                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

                    term3 = 0

                    
                if (Ocean_boundary[j,ip1] == 5 or Ocean_boundary[j,ip1] == 0 or Ocean_boundary[j,ip1] == 2) and (Ocean_boundary[j,i-1] == 1 or Ocean_boundary[j,i-1] == 3) and (Ocean_boundary[j+1,i] == 1 or Ocean_boundary[j+1,i] == 3) and (Ocean_boundary[j-1,i] == 1 or Ocean_boundary[j-1,i] == 3): 
                   factor1 = csc2[j - 1] / (h ** 2)
                   term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j,i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

                   factor2 = 1 / (h ** 2)
                   term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i])

                   term3 = 0
                   
                if (Ocean_boundary[j,ip1] == 1 or Ocean_boundary[j,ip1] == 3) and (Ocean_boundary[j,i-1] == 5 or Ocean_boundary[j,i-1] == 0 or Ocean_boundary[j,i-1] == 2) and (Ocean_boundary[j+1,i] == 1 or Ocean_boundary[j+1,i] == 3) and (Ocean_boundary[j-1,i] == 1 or Ocean_boundary[j-1,i] == 3):    
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

                    term3 = 0
                
                if (Ocean_boundary[j,ip1] == 1 or Ocean_boundary[j,ip1] == 3) and (Ocean_boundary[j,i-1] == 1 or Ocean_boundary[j,i-1] == 3) and (Ocean_boundary[j+1,i] == 5 or Ocean_boundary[j+1,i] == 0 or Ocean_boundary[j+1,i] == 2) and (Ocean_boundary[j-1,i] == 1 or Ocean_boundary[j-1,i] == 3):
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j , i])
                    
                if (Ocean_boundary[j,ip1] == 1 or Ocean_boundary[j,ip1] == 3) and (Ocean_boundary[j,i-1] == 1 or Ocean_boundary[j,i-1] == 3) and (Ocean_boundary[j+1,i] == 1 or Ocean_boundary[j+1,i] == 3) and (Ocean_boundary[j-1,i] == 5 or Ocean_boundary[j-1,i] == 0 or Ocean_boundary[j-1,i] == 2):
                     factor1 = csc2[j - 1] / (h ** 2)
                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

                     factor2 = 1 / (h ** 2)
                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

                     term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j , i] - temperature[j - 1, i])
                     
                if (Ocean_boundary[j,ip1] == 5 or Ocean_boundary[j,ip1] == 0 or Ocean_boundary[j,ip1] == 2) and (Ocean_boundary[j,i-1] == 5 or Ocean_boundary[j,i-1] == 0 or Ocean_boundary[j,i-1] == 2) and (Ocean_boundary[j+1,i] == 1 or Ocean_boundary[j+1,i] == 3) and (Ocean_boundary[j-1,i] == 1 or Ocean_boundary[j-1,i] == 3):
                    
                   factor1 = csc2[j - 1] / (h ** 2)
                   term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

                   factor2 = 1 / (h ** 2)
                   term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

                   term3 = 0

                if (Ocean_boundary[j,ip1] == 5 or Ocean_boundary[j,ip1] == 0 or Ocean_boundary[j,ip1] == 2) and (Ocean_boundary[j,i-1] == 1 or Ocean_boundary[j,i-1] == 3) and (Ocean_boundary[j+1,i] == 5 or Ocean_boundary[j+1,i] == 0 or Ocean_boundary[j+1,i] == 2)  and (Ocean_boundary[j-1,i] == 1 or Ocean_boundary[j-1,i] == 3):
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j , i])

                    
                if (Ocean_boundary[j,ip1] == 5 or Ocean_boundary[j,ip1] == 0 or Ocean_boundary[j,ip1] == 2) and (Ocean_boundary[j,i-1] == 1 or Ocean_boundary[j,i-1] == 3) and (Ocean_boundary[j+1,i] == 1 or Ocean_boundary[j+1,i] == 3) and (Ocean_boundary[j-1,i] == 5 or Ocean_boundary[j-1,i] == 0 or Ocean_boundary[j-1,i] == 2):
                   factor1 = csc2[j - 1] / (h ** 2)
                   term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

                   factor2 = 1 / (h ** 2)
                   term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

                   term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j , i] - temperature[j - 1, i])

                if (Ocean_boundary[j,ip1] == 1 or Ocean_boundary[j,ip1] == 3) and (Ocean_boundary[j,i-1] == 5 or Ocean_boundary[j,i-1] == 0 or Ocean_boundary[j,i-1] == 2) and (Ocean_boundary[j+1,i] == 5 or Ocean_boundary[j+1,i] == 0 or Ocean_boundary[j+1,i] == 2) and (Ocean_boundary[j-1,i] == 1 or Ocean_boundary[j-1,i] == 3):
                   factor1 = csc2[j - 1] / (h ** 2)
                   term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

                   factor2 = 1 / (h ** 2)
                   term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

                   term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j, i])

                if (Ocean_boundary[j,ip1] == 1 or Ocean_boundary[j,ip1] == 3) and (Ocean_boundary[j,i-1] == 5 or Ocean_boundary[j,i-1] == 0 or Ocean_boundary[j,i-1] == 2) and (Ocean_boundary[j+1,i] == 1 or Ocean_boundary[j+1,i] == 3) and (Ocean_boundary[j-1,i] == 5 or Ocean_boundary[j-1,i] == 0 or Ocean_boundary[j-1,i] == 2):
                   factor1 = csc2[j - 1] / (h ** 2)
                   term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

                   factor2 = 1 / (h ** 2)
                   term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i])

                   term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j, i] - temperature[j - 1, i])

     
                if (Ocean_boundary[j,ip1] == 1 or Ocean_boundary[j,ip1] == 3) and (Ocean_boundary[j,i-1] == 1 or Ocean_boundary[j,i-1] == 3) and (Ocean_boundary[j+1,i] == 5 or Ocean_boundary[j+1,i] == 0 or Ocean_boundary[j+1,i] == 2) and (Ocean_boundary[j-1,i] == 5 or Ocean_boundary[j-1,i] == 0 or Ocean_boundary[j-1,i] == 2):
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])

                if (Ocean_boundary[j,ip1] == 5 or Ocean_boundary[j,ip1] == 0 or Ocean_boundary[j,ip1] == 2) and (Ocean_boundary[j,i-1] == 5 or Ocean_boundary[j,i-1] == 0 or Ocean_boundary[j,i-1] == 2) and (Ocean_boundary[j+1,i] == 5 or Ocean_boundary[j+1,i] == 0 or Ocean_boundary[j+1,i] == 2) and (Ocean_boundary[j-1,i] == 1 or Ocean_boundary[j-1,i] == 3):
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j , i])

                                 
                if (Ocean_boundary[j,ip1] == 1 or Ocean_boundary[j,ip1] == 3) and (Ocean_boundary[j,i-1] == 5  or Ocean_boundary[j,i-1] == 0 or Ocean_boundary[j,i-1] == 2) and (Ocean_boundary[j+1,i] == 5 or Ocean_boundary[j+1,i] == 0 or Ocean_boundary[j+1,i] == 2) and (Ocean_boundary[j-1,i] == 5 or Ocean_boundary[j-1,i] == 0 or Ocean_boundary[j-1,i] == 2):
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])

                if (Ocean_boundary[j,ip1] == 5 or Ocean_boundary[j,ip1] == 0 or Ocean_boundary[j,ip1] == 2) and (Ocean_boundary[j,i-1] == 1 or Ocean_boundary[j,i-1] == 3) and (Ocean_boundary[j+1,i] == 5 or Ocean_boundary[j+1,i] == 0 or Ocean_boundary[j+1,i] == 2)  and (Ocean_boundary[j-1,i] == 5 or Ocean_boundary[j-1,i] == 0 or Ocean_boundary[j-1,i] == 2):
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])

                if (Ocean_boundary[j,ip1] == 5 or Ocean_boundary[j,ip1] == 0 or Ocean_boundary[j,ip1] == 2) and (Ocean_boundary[j,i-1] == 5 or Ocean_boundary[j,i-1] == 0 or Ocean_boundary[j,i-1] == 2) and (Ocean_boundary[j+1,i] == 1 or Ocean_boundary[j+1,i] == 3) and (Ocean_boundary[j-1,i] == 5 or Ocean_boundary[j-1,i] == 0 or Ocean_boundary[j-1,i] == 2):
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j , i] - temperature[j - 1, i])

                    
                if (Ocean_boundary[j,ip1] == 5 or Ocean_boundary[j,ip1] == 0 or Ocean_boundary[j,ip1] == 2) and (Ocean_boundary[j,i-1] == 5 or Ocean_boundary[j,i-1] == 0  or Ocean_boundary[j,i-1] == 2) and (Ocean_boundary[j+1,i] == 5 or Ocean_boundary[j+1,i] == 0 or Ocean_boundary[j+1,i] == 2) and (Ocean_boundary[j-1,i] == 5 or Ocean_boundary[j-1,i] == 0 or Ocean_boundary[j-1,i] == 2): 
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])

            elif (Ocean_boundary[j,i] == 1 or Ocean_boundary[j,i] == 3):   
                term1 = np.nan
                term2 = np.nan
                term3 = np.nan
            else:    
                # Note that csc2 does not contain the values at the poles
                factor1 = csc2[j - 1] /(h ** 2)
                term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1]))  * temperature[j, ip1])
    
                factor2 = 1 / (h ** 2)
                term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i]))  * temperature[j + 1, i])
    
                term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])

            result[j, i] = term1 + term2 + term3
            

    return result



# @njit    
# def calc_diffusion_operator_inner_land(h, area, n_latitude, n_longitude, csc2, cot, diffusion_coeff, temperature, Surface_boundary):
#     result = np.zeros(diffusion_coeff.shape)

#     # North Pole
#     factor = np.sin(h / 2) / (4 * np.pi * area[0])
#     #result[0, :] = factor * 0.5 * np.dot(diffusion_coeff[0, :] + diffusion_coeff[1, :], temperature[1, :] - temperature[0, :])
#     result[0, :] = np.ones(128)  * np.nan
#     # South Pole
#     factor = np.sin(h / 2) / (4 * np.pi * area[-1])
#     result[-1, :] = factor * 0.5 * np.dot(diffusion_coeff[-1, :] + diffusion_coeff[-2, :],temperature[-2, :] - temperature[-1, :])
#     #result[-1, :] = np.zeros(128) #da am Südpol nur Land ist --> keine Ozean Diffusion
    
#     for i in range(n_longitude):
#         # Only loop over inner nodes
#         for j in range(1, n_latitude - 1):
#             # There are the special cases of i=0 and i=n_longitude-1.
#             # We have a periodic boundary condition, so for i=0, we want i-1 to be the last entry.
#             # This happens automatically in Python when i=-1.
#             # For i=n_longitude-1, we want i+1 to be 0.
#             # For this, we define a variable ip1 (i plus 1) to avoid duplicating code.
#             if i == n_longitude - 1:
#                 ip1 = 0
#             else:
#                 ip1 = i + 1
            
#             if Surface_boundary[j,i] == 0 :  #and i == k and j == l
#                 if (Surface_boundary[j,ip1] == 5 or  Surface_boundary[j,ip1] == 2 ) and (Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2) and (Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2) and (Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2):
#                     factor1 = csc2[j - 1] / (h ** 2)
#                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])
                    
#                     factor2 = 1 / (h ** 2)
#                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

#                     term3 = 0

                    
#                 if (Surface_boundary[j,ip1] == 1 or Surface_boundary[j,ip1] == 0 or Surface_boundary[j,ip1] == 3) and (Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2) and (Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2) and (Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2): 
#                     factor1 = csc2[j - 1] / (h ** 2)
#                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j,i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

#                     factor2 = 1 / (h ** 2)
#                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i])

#                     term3 = 0
                   
#                 if (Surface_boundary[j,ip1] == 5 or Surface_boundary[j,ip1] == 2) and (Surface_boundary[j,i-1] == 1 or Surface_boundary[j,i-1] == 0 or Surface_boundary[j,i-1] == 3) and (Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2) and (Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2):    
#                     factor1 = csc2[j - 1] / (h ** 2)
#                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

#                     factor2 = 1 / (h ** 2)
#                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

#                     term3 = 0
                
#                 if (Surface_boundary[j,ip1] == 5 or Surface_boundary[j,ip1] == 2) and (Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2) and (Surface_boundary[j+1,i] == 1 or Surface_boundary[j+1,i] == 0 or Surface_boundary[j+1,i] == 3) and (Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2):
#                     factor1 = csc2[j - 1] / (h ** 2)
#                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

#                     factor2 = 1 / (h ** 2)
#                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

#                     term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j , i])
                    
#                 if (Surface_boundary[j,ip1] == 5 or Surface_boundary[j,ip1] == 2) and (Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2) and (Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2) and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 0 or Surface_boundary[j-1,i] == 3):
#                       factor1 = csc2[j - 1] / (h ** 2)
#                       term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

#                       factor2 = 1 / (h ** 2)
#                       term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

#                       term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j , i] - temperature[j - 1, i])
                     
#                 if (Surface_boundary[j,ip1] == 1 or Surface_boundary[j,ip1] == 0 or Surface_boundary[j,ip1] == 3) and (Surface_boundary[j,i-1] == 1 or Surface_boundary[j,i-1] == 0 or Surface_boundary[j,i-1] == 3) and (Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2) and (Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2):
                    
#                     factor1 = csc2[j - 1] / (h ** 2)
#                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

#                     factor2 = 1 / (h ** 2)
#                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

#                     term3 = 0

#                 if (Surface_boundary[j,ip1] == 1 or Surface_boundary[j,ip1] == 0 or Surface_boundary[j,ip1] == 3) and (Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2) and (Surface_boundary[j+1,i] == 1 or Surface_boundary[j+1,i] == 0 or Surface_boundary[j+1,i] == 3)  and (Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2):
#                     factor1 = csc2[j - 1] / (h ** 2)
#                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

#                     factor2 = 1 / (h ** 2)
#                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

#                     term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j , i])

                    
#                 if (Surface_boundary[j,ip1] == 1 or Surface_boundary[j,ip1] == 0 or Surface_boundary[j,ip1] == 3) and (Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2) and (Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2) and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 0 or Surface_boundary[j-1,i] == 3):
#                     factor1 = csc2[j - 1] / (h ** 2)
#                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

#                     factor2 = 1 / (h ** 2)
#                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

#                     term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j , i] - temperature[j - 1, i])

#                 if (Surface_boundary[j,ip1] == 5 or Surface_boundary[j,ip1] == 2) and (Surface_boundary[j,i-1] == 1 or Surface_boundary[j,i-1] == 0 or Surface_boundary[j,i-1] == 3) and (Surface_boundary[j+1,i] == 1 or Surface_boundary[j+1,i] == 0 or Surface_boundary[j+1,i] == 3) and (Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2):
#                     factor1 = csc2[j - 1] / (h ** 2)
#                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

#                     factor2 = 1 / (h ** 2)
#                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

#                     term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j, i])

#                 if (Surface_boundary[j,ip1] == 5 or Surface_boundary[j,ip1] == 2) and (Surface_boundary[j,i-1] == 1 or Surface_boundary[j,i-1] == 0 or Surface_boundary[j,i-1] == 3) and (Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2) and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 0 or Surface_boundary[j-1,i] == 3):
#                     factor1 = csc2[j - 1] / (h ** 2)
#                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

#                     factor2 = 1 / (h ** 2)
#                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i])

#                     term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j, i] - temperature[j - 1, i])

     
#                 if (Surface_boundary[j,ip1] == 5 or Surface_boundary[j,ip1] == 2) and (Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2) and (Surface_boundary[j+1,i] == 1 or Surface_boundary[j+1,i] == 0 or Surface_boundary[j+1,i] == 3) and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 0 or Surface_boundary[j-1,i] == 3):
#                     factor1 = csc2[j - 1] / (h ** 2)
#                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

#                     factor2 = 1 / (h ** 2)
#                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

#                     term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])

#                 if (Surface_boundary[j,ip1] == 1 or Surface_boundary[j,ip1] == 0 or Surface_boundary[j,ip1] == 3) and (Surface_boundary[j,i-1] == 1 or Surface_boundary[j,i-1] == 0 or Surface_boundary[j,i-1] == 3) and (Surface_boundary[j+1,i] == 1 or Surface_boundary[j+1,i] == 0 or Surface_boundary[j+1,i] == 3) and (Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2):
#                     factor1 = csc2[j - 1] / (h ** 2)
#                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

#                     factor2 = 1 / (h ** 2)
#                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

#                     term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j , i])

                                 
#                 if (Surface_boundary[j,ip1] == 5 or Surface_boundary[j,ip1] == 2) and (Surface_boundary[j,i-1] == 1  or Surface_boundary[j,i-1] == 0 or Surface_boundary[j,i-1] == 3) and (Surface_boundary[j+1,i] == 1 or Surface_boundary[j+1,i] == 0 or Surface_boundary[j+1,i] == 3) and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 0 or Surface_boundary[j-1,i] == 3):
#                     factor1 = csc2[j - 1] / (h ** 2)
#                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

#                     factor2 = 1 / (h ** 2)
#                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

#                     term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])

#                 if (Surface_boundary[j,ip1] == 1 or Surface_boundary[j,ip1] == 0 or Surface_boundary[j,ip1] == 3) and (Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2) and (Surface_boundary[j+1,i] == 1 or Surface_boundary[j+1,i] == 0 or Surface_boundary[j+1,i] == 3)  and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 0 or Surface_boundary[j-1,i] == 3):
#                     factor1 = csc2[j - 1] / (h ** 2)
#                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

#                     factor2 = 1 / (h ** 2)
#                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

#                     term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])

#                 if (Surface_boundary[j,ip1] == 1 or Surface_boundary[j,ip1] == 0 or Surface_boundary[j,ip1] == 3) and (Surface_boundary[j,i-1] == 1 or Surface_boundary[j,i-1] == 0 or Surface_boundary[j,i-1] == 3) and (Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2) and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 0 or Surface_boundary[j-1,i] == 3):
#                     factor1 = csc2[j - 1] / (h ** 2)
#                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

#                     factor2 = 1 / (h ** 2)
#                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

#                     term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j , i] - temperature[j - 1, i])

                    
#                 if (Surface_boundary[j,ip1] == 1 or Surface_boundary[j,ip1] == 0 or Surface_boundary[j,ip1] == 3) and (Surface_boundary[j,i-1] == 1 or Surface_boundary[j,i-1] == 0  or Surface_boundary[j,i-1] == 3) and (Surface_boundary[j+1,i] == 1 or Surface_boundary[j+1,i] == 0 or Surface_boundary[j+1,i] == 3) and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 0 or Surface_boundary[j-1,i] == 3): 
#                     factor1 = csc2[j - 1] / (h ** 2)
#                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

#                     factor2 = 1 / (h ** 2)
#                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

#                     term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])

#             elif (Surface_boundary[j,i] == 2 or Surface_boundary[j,i] == 5):   
#                 term1 = np.nan
#                 term2 = np.nan
#                 term3 = np.nan
                
#             else:    
#                 # Note that csc2 does not contain the values at the poles
#                 factor1 = csc2[j - 1] /(h ** 2)
#                 term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1]))  * temperature[j, ip1])
    
#                 factor2 = 1 / (h ** 2)
#                 term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i]))  * temperature[j + 1, i])
    
#                 term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])

#             result[j, i] = term1 + term2 + term3
            

#     return result



def timestep_euler_forward_SNOW(mesh,T_S_land, T_ATM, solar_forcing, SNOW, t, delta_t): #calc new snow thickness 
    
    SNOW_new = SNOW - delta_t * (1/mesh.Lf * (-mesh.A_up - mesh.B_up * T_S_land + mesh.A_dn + mesh.B_dn * T_ATM + solar_forcing) * (SNOW >0))
    
    return SNOW_new


def FreezeAndMelt_Snow(T_S_land, mesh, SNOW, P_ocn): #calc new ice distribution

       T_S_land_new = copy.copy(T_S_land)
       SNOW_new = copy.copy(SNOW)
       z = mesh.Lf/(mesh.C_s*350)
      
       for i in range(mesh.n_longitude):
         for j in range(mesh.n_latitude):   
            if SNOW[j,i] < 0:
              
              SNOW_new[j,i] = 0
              T_S_land_new[j,i] = T_S_land[j,i] - z* SNOW[j,i]
              
              if T_S_land_new[j,i] < mesh.Tf:
                  SNOW_new[j,i] = (mesh.Tf-T_S_land_new[j,i])/z
                  T_S_land_new[j,i] = mesh.Tf
           
            elif( SNOW[j,i] == 0 and T_S_land[j,i] < mesh.Tf):
              
                  SNOW_new[j,i] = (mesh.Tf-T_S_land[j,i])/z
                  T_S_land_new[j,i] = mesh.Tf
          
              
            elif SNOW[j,i] > 0:
              SNOW_new[j,i] = SNOW[j,i] + (mesh.Tf-T_S_land[j,i])/z
              T_S_land_new[j,i] = mesh.Tf
         
              if SNOW_new[j,i] < 0:
                  T_S_land_new[j,i] = mesh.Tf -z*SNOW_new[j,i]
                  SNOW_new[j,i] = 0
        
    
       return T_S_land_new, SNOW_new
   
    
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