#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 08:43:09 2023

@author: ricij
"""

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy import special
import copy

def unite_surface_temp(T_S_OCN, T_S_land, mesh, Geo_dat): #unite the surface temperature of the ocean and the land in one variable
    
    Surface_Temp = np.zeros((mesh.n_latitude,mesh.n_longitude))
    for i in range(mesh.n_latitude):
        for j in range(mesh.n_longitude):
                if Geo_dat[i,j] == 1 or Geo_dat[i,j] == 3: 
                    Surface_Temp[i,j] = T_S_land[i,j]
                else: 
                    Surface_Temp[i,j] = T_S_OCN[i,j]
                    
    return Surface_Temp

def LandDstr_wLakes(Geo_dat):
    dstr = copy.copy(Geo_dat)
    dstr[6,23] = 4
    dstr[6,24] = 4
    dstr[5,26] = 4
    dstr[6,26] = 4
    dstr[6,27] = 4
    dstr[7,28] = 4
    
    dstr[6,31] = 4
    dstr[7,31] = 4
    dstr[8,35] = 4
    dstr[8,36] = 4
    dstr[9,36] = 4
    
    dstr[10,31] = 4
    dstr[10,32] = 4
    dstr[10,33] = 4
    dstr[10,34] = 4
    
    dstr[11,30] = 4
    dstr[11,31] = 4
    dstr[11,32] = 4
    dstr[11,33] = 4
    dstr[11,34] = 4
    
    dstr[12,33] = 4
    dstr[12,34] = 4
    dstr[12,35] = 4
    dstr[13,34] = 4
    
    dstr[15,32] = 4
    dstr[15,40] = 4
    dstr[15,41] = 4
    
    dstr[14,63] = 4
    dstr[19,62] = 4
    dstr[19,63] = 4
    dstr[12,67] = 4
    
    dstr[9,71] = 4
    dstr[10,70] = 4
    dstr[11,70] = 4
    dstr[12,70] = 4
    dstr[12,69] = 4
    dstr[11,71] = 4
    
    dstr[17,65] = 4
    dstr[17,66] = 4
    dstr[17,67] = 4
    dstr[17,69] = 4
    
    dstr[18,64] = 4
    dstr[18,65] = 4
    dstr[18,66] = 4
    dstr[18,67] = 4
    dstr[18,68] = 4
    dstr[18,70] = 4
    
    dstr[19,67] = 4
    dstr[19,68] = 4
    dstr[19,69] = 4
    dstr[19,70] = 4
    dstr[19,71] = 4
    dstr[19,72] = 4
    
    dstr[20,68] = 4
    dstr[20,69] = 4
    dstr[20,70] = 4
    dstr[20,71] = 4
    dstr[20,72] = 4
    dstr[20,73] = 4
    dstr[20,74] = 4
    dstr[20,75] = 4
    
    dstr[16,74] = 4
    dstr[17,74] = 4
    dstr[16,75] = 4
    dstr[17,73] = 4
    dstr[17,76] = 4
    dstr[17,77] = 4
    
    dstr[16,80] = 4
    dstr[16,81] = 4
    dstr[17,81] = 4
    dstr[18,81] = 4
    dstr[18,82] = 4
    
    dstr[25,101] = 4
    dstr[22,81] = 4
    dstr[23,82] = 4
    dstr[23,76] = 4
    dstr[24,76] = 4
    dstr[25,77] = 4
    
    dstr[26,78] = 4
    dstr[27,78] = 4
    dstr[28,79] = 4
    dstr[32,75] = 4
    return dstr

def read_geography(filepath):
    return np.genfromtxt(filepath, dtype=np.int8)

def robinson_projection(nlatitude, nlongitude):
    def x_fun(lon, lat):
        return lon / np.pi * (0.0379 * lat ** 6 - 0.15 * lat ** 4 - 0.367 * lat ** 2 + 2.666)

    def y_fun(_, lat):
        return 0.96047 * lat - 0.00857 * np.sign(lat) * np.abs(lat) ** 6.41

    # Longitude goes from -pi to pi (not included), latitude from -pi/2 to pi/2.
    # Latitude goes backwards because the data starts in the North, which corresponds to a latitude of pi/2.
    x_lon = np.linspace(-np.pi, np.pi, nlongitude, endpoint=False)
    y_lat = np.linspace(np.pi / 2, -np.pi / 2, nlatitude)

    x = np.array([[x_fun(lon, lat) for lon in x_lon] for lat in y_lat])
    y = np.array([[y_fun(lon, lat) for lon in x_lon] for lat in y_lat])

    return x, y


# Plot data at grid points in Robinson projection. Return the colorbar for customization.
# This will be reused in other milestones.
def plot_robinson_projection(data, title, plot_continents=False, geo_dat=[], **kwargs):
    # Get the coordinates for the Robinson projection.
    nlatitude, nlongitude = data.shape
    x, y = robinson_projection(nlatitude, nlongitude)

    # Start plotting.
    fig, ax = plt.subplots()

    # Create contour plot of geography information against x and y.
    im = ax.contourf(x, y, data, **kwargs)
    if plot_continents:
        ax.contour(x,y,geo_dat,colors='black',linewidths=0.25, linestyles='solid')
    plt.title(title)
    ax.set_aspect("equal")

    # Remove axes and ticks.
    plt.xticks([])
    plt.yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Colorbar with the same height as the plot. Code copied from
    # https://stackoverflow.com/a/18195921
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    return cbar

def read_true_longitude(filepath):
    return np.genfromtxt(filepath, dtype=np.float64) #from lecture


def calc_radiative_cooling_co2(co2_concentration, co2_concentration_base=315.0,  
                               radiative_cooling_base=210.3):
    #return radiative_cooling_base - 5.35 * np.log(co2_concentration / co2_concentration_base)  #from lecture
    return  5.35 * np.log(co2_concentration / co2_concentration_base)  #from lecture

def calc_coalbedo(geo_dat, phi_i_n, phi_i_s):
    def legendre(latitude):
        return 0.5 * (3 * np.sin(latitude) ** 2 - 1)
    
    def coalbedo_ocn(latitude, phi_i_n, phi_i_s): #calculation in the paper 
        a0 = 0.72
        ai = 0.36
        a2 = (a0 - ai)/((np.pi/2)**2)
        
        if latitude > 0:
            coalbedo =  0.5 * ((a0-a2*latitude**2 + ai) - (a0-a2*latitude**2 - ai) * special.erf((latitude-phi_i_n)/0.04)) #North
        else:
            coalbedo = 0.5 * ((a0-a2*latitude**2 + ai) - (a0-a2*latitude**2 - ai) * special.erf((phi_i_s-latitude)/0.04)) #South
        
        return coalbedo     
        

    def coalbedo(surface_type, latitude, phi_i_n, phi_i_s):
        if surface_type == 1:
            return 1-(0.3 + 0.12 * legendre(latitude))
        elif surface_type == 2:
            #return 0.4
            return coalbedo_ocn(latitude, phi_i_n, phi_i_s)
        elif surface_type == 3:
            return 0.25
        elif surface_type == 5:
            #return 1-(0.29 + 0.12 * legendre(latitude)) #coalbedo in lecture 
            return coalbedo_ocn(latitude, phi_i_n,phi_i_s)
        else:
            raise ValueError(f"Unknown surface type {surface_type}.")

    nlatitude, nlongitude = geo_dat.shape
    y_lat = np.linspace(np.pi/2,- np.pi/2, nlatitude)

    # Map surface type to albedo.
    return np.array([[coalbedo(geo_dat[i, j], y_lat[i], phi_i_n[j], phi_i_s[j])
                      for j in range(nlongitude)]
                     for i in range(nlatitude)])

 
    
def change_geo_dat(Geo_dat, H_I, mesh): #change in sea ice distribution due to change in temperature
   
    for  j in range(mesh.n_longitude):
        for i in  range(mesh.n_latitude):
            if H_I[i,j] > 0 and Geo_dat[i,j] == 5: 
                Geo_dat[i,j] = 2
                
            if H_I[i,j] <= 0 and Geo_dat[i,j] == 2: 
                Geo_dat[i,j] = 5
                
     
    return Geo_dat 

def Get_Ocean_Land_distribution(Geo_dat): 
    latitude, longitude = Geo_dat.shape
    OCN_LAND_DSTB = copy.copy(Geo_dat)
    for i in range(latitude):
        for j in range(longitude):
            if Geo_dat[i,j] == 2: 
                OCN_LAND_DSTB[i,j] = 5
                
    return OCN_LAND_DSTB  

def Get_Ocean_Boundary_Distribution(Geo_dat):      
    latitude, longitude = Geo_dat.shape
    OCN_BND = copy.copy(Geo_dat)
    for i in range(1,latitude-1):
        for j in range(longitude):
            if j == longitude - 1:
                jp1 = 0
            else:
                jp1 = j + 1
            if (Geo_dat[i,j]  == 5) or (Geo_dat[i,j] == 2): 
                if (Geo_dat[i+1,j] == 3) or  (Geo_dat[i+1,j] == 1):
                    OCN_BND[i,j] = 0
                if (Geo_dat[i-1,j] == 3) or  (Geo_dat[i-1,j] == 1):
                     OCN_BND[i,j] = 0
                if (Geo_dat[i,jp1] == 3) or  (Geo_dat[i,jp1] == 1):
                     OCN_BND[i,j] = 0     
                if (Geo_dat[i,j-1] == 3) or  (Geo_dat[i,j-1] == 1):
                     OCN_BND[i,j] = 0   
                     
    return OCN_BND

def Get_Surface_Boundary_Distribution(Geo_dat):
    latitude, longitude = Geo_dat.shape
    Surface_BND = copy.copy(Geo_dat)
    for i in range(1,latitude-1):
        for j in range(longitude):
            if j == longitude - 1:
                jp1 = 0
            else:
                jp1 = j + 1
            if (Geo_dat[i,j]  == 1) or (Geo_dat[i,j] == 3): 
                if (Geo_dat[i+1,j] == 5) or  (Geo_dat[i+1,j] == 2):
                    Surface_BND[i,j] = 0
                if (Geo_dat[i-1,j] == 5) or  (Geo_dat[i-1,j] == 2):
                     Surface_BND[i,j] = 0
                if (Geo_dat[i,jp1] == 5) or  (Geo_dat[i,jp1] == 2):
                     Surface_BND[i,j] = 0     
                if (Geo_dat[i,j-1] == 5) or  (Geo_dat[i,j-1] == 2):
                     Surface_BND[i,j] = 0   
                     
    return Surface_BND



def calc_heat_capacity(T): #from lecture
    n_latitude, n_longitude = T.shape  
    C = np.zeros((n_latitude,n_longitude))
    
    C_atm = 1.225*1000*3850
    C_mixed = 1030 *4000*70
    C_ice = 917*2000*1.5
    C_soil = 1350*750*1
    C_snow = 400*880*0.5
    sek_per_year = 3.15576*10**7
    
    for i in range(0,n_latitude):
        for j in range(0,n_longitude):
            if T[i,j] == 1: 
                C[i,j] = (C_soil + C_atm)/sek_per_year
            if T[i,j] == 2: 
                C[i,j] = (C_ice + C_atm)/sek_per_year
            if T[i,j] == 3: 
                C[i,j] = (C_snow + C_atm)/sek_per_year
            if T[i,j] == 5: 
                C[i,j] = (C_mixed + C_atm)/sek_per_year
                   
    return C

def calc_diffusion_coefficients(geo_dat): #from lecture
    nlatitude, nlongitude = geo_dat.shape

    coeff_ocean_poles = 0.40
    coeff_ocean_equator = 0.65
    coeff_equator = 0.65
    coeff_north_pole = 0.28
    coeff_south_pole = 0.20

    def diffusion_coefficient(j, i):
        # Compute the j value of the equator
        j_equator = int(nlatitude / 2)

        theta = np.pi * j / np.real(nlatitude - 1)
        colat = np.sin(theta) ** 5

        geo = geo_dat[j, i]
        if geo == 5:  # ocean
            return coeff_ocean_poles + (coeff_ocean_equator - coeff_ocean_poles) * colat
        else:  # land, sea ice, etc
            if j <= j_equator:  # northern hemisphere
                # on the equator colat=1 -> coefficients for norhern/southern hemisphere cancels out
                return coeff_north_pole + (coeff_equator - coeff_north_pole) * colat
            else:  # southern hemisphere
                return coeff_south_pole + (coeff_equator - coeff_south_pole) * colat

    return np.array([[diffusion_coefficient(j, i) for i in range(nlongitude)] for j in range(nlatitude)])

def insolation(latitude, true_longitude, solar_constant, eccentricity,
               obliquity, precession_distance):
    # Determine if there is no sunset or no sunrise.
    sin_delta = np.sin(obliquity) * np.sin(true_longitude)
    cos_delta = np.sqrt(1 - sin_delta ** 2)
    tan_delta = sin_delta / cos_delta

    # Note that z can be +-infinity.
    # This is not a problem, as it is only used for the comparison with +-1.
    # We will never enter the `else` case below if z is +-infinity.
    z = -np.tan(latitude) * tan_delta

    if z >= 1:
        # Latitude where there is no sunrise
        return 0.0
    else:
        rho = ((1 - eccentricity * np.cos(true_longitude - precession_distance))
               / (1 - eccentricity ** 2)) ** 2

        if z <= -1:
            # Latitude where there is no sunset
            return solar_constant * rho * np.sin(latitude) * sin_delta
        else:
            h0 = np.arccos(z)
            second_term = h0 * np.sin(latitude) * sin_delta + np.cos(latitude) * cos_delta * np.sin(h0)
            return solar_constant * rho / np.pi * second_term
   
def calc_insolation(y_lat, true_longitudes, solar_constant=1371.685,    
                       eccentricity= 0.016740, obliquity=0.409253,
                       precession_distance=1.783037):
    nlatitude = y_lat.size
    
    return np.array([[insolation(y_lat[j], true_longitude, solar_constant, eccentricity,
                   obliquity, precession_distance)
                       for true_longitude in true_longitudes]
                     for j in range(nlatitude)])


def calc_solar_forcing(insolation, coalbedo, mesh):
    solar_forcing = np.zeros((mesh.n_latitude, mesh.n_longitude))
    for i in range(mesh.n_longitude):
        solar_forcing[:,i]  = insolation * coalbedo[:,i]
    return solar_forcing    

def plot_solar_forcing(solar_forcing, timestep, show_plot=False):
    vmin = np.amin(solar_forcing)
    vmax = np.amax(solar_forcing) * 1.05
    levels = np.linspace(vmin, vmax, 200)

    # Reuse plotting function from milestone 1.
    ntimesteps = solar_forcing.shape[2]
    day = (np.int_(timestep / ntimesteps * 365) + 80) % 365
    cbar = plot_robinson_projection(solar_forcing[:, :, timestep],
                                    f"Solar Forcing for Day {day}",
                                    levels=levels, cmap="gist_heat",
                                    vmin=vmin, vmax=vmax)
    cbar.set_label("solar forcing")

    # Adjust size of plot to viewport to prevent clipping of the legend.
    plt.tight_layout()

    filename = 'solar_forcing_{}.png'.format(timestep)
    plt.savefig(filename, dpi=300)
    if show_plot:
        plt.show()
    plt.close()

    return filename


def calc_diffusion_operator_ocn(mesh, diffusion_coeff, temperature, Ocean_boundary, k, l):
    h = mesh.h
    area = mesh.area
    n_latitude = mesh.n_latitude
    n_longitude = mesh.n_longitude
    csc2 = mesh.csc2
    cot = mesh.cot

    return calc_diffusion_operator_inner_ocn(h, area, n_latitude, n_longitude, csc2, cot, diffusion_coeff, temperature, Ocean_boundary, k, l )

@njit    
def calc_diffusion_operator_inner_ocn(h, area, n_latitude, n_longitude, csc2, cot, diffusion_coeff, temperature, Ocean_boundary, k, l):
    result = np.zeros(diffusion_coeff.shape)

    # North Pole
    factor = np.sin(h / 2) / (4 * np.pi * area[0])
    result[0, :] = factor * 0.5 * np.dot(diffusion_coeff[0, :] + diffusion_coeff[1, :], temperature[1, :] - temperature[0, :])

    # South Pole
    # factor = np.sin(h / 2) / (4 * np.pi * area[-1])
    # result[-1, :] = factor * 0.5 * np.dot(diffusion_coeff[-1, :] + diffusion_coeff[-2, :],temperature[-2, :] - temperature[-1, :])
    #result[-1, :] = np.zeros(128) #da am Südpol nur Land ist --> keine Ozean Diffusion
    result[-1, :] = np.ones(128) * np.nan
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
            
            if Ocean_boundary[j,i] == 0 :  #and i == k and j == l
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

def calc_diffusion_operator_atm(mesh, diffusion_coeff, temperature):
    h = mesh.h
    area = mesh.area
    n_latitude = mesh.n_latitude
    n_longitude = mesh.n_longitude
    csc2 = mesh.csc2
    cot = mesh.cot

    return calc_diffusion_operator_inner_atm(h, area, n_latitude, n_longitude, csc2, cot, diffusion_coeff, temperature)

@njit    
def calc_diffusion_operator_inner_atm(h, area, n_latitude, n_longitude, csc2, cot, diffusion_coeff, temperature):
    result = np.zeros(diffusion_coeff.shape)

    # North Pole
    factor = np.sin(h / 2) / (4 * np.pi * area[0])
    result[0, :] = factor * 0.5 * np.dot(diffusion_coeff[0, :] + diffusion_coeff[1, :], temperature[1, :] - temperature[0, :])

    # South Pole
    factor = np.sin(h / 2) / (4 * np.pi * area[-1])
    result[-1, :] = factor * 0.5 * np.dot(diffusion_coeff[-1, :] + diffusion_coeff[-2, :],temperature[-2, :] - temperature[-1, :])
   
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
                               (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) *
                               temperature[j - 1, i]
                               + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) *
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
    result[-1, :] = factor * 0.5 * np.dot(diffusion_coeff[-1, :] + diffusion_coeff[-2, :],temperature[-2, :] - temperature[-1, :])
    #result[-1, :] = np.zeros(128) #da am Südpol nur Land ist --> keine Ozean Diffusion
    
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
            
            if Surface_boundary[j,i] == 0 :  #and i == k and j == l
                if (Surface_boundary[j,ip1] == 5 or  Surface_boundary[j,ip1] == 2 ) and (Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2) and (Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2) and (Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2):
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])
                    
                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

                    term3 = 0

                    
                if (Surface_boundary[j,ip1] == 1 or Surface_boundary[j,ip1] == 0 or Surface_boundary[j,ip1] == 3) and (Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2) and (Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2) and (Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2): 
                   factor1 = csc2[j - 1] / (h ** 2)
                   term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j,i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

                   factor2 = 1 / (h ** 2)
                   term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i])

                   term3 = 0
                   
                if (Surface_boundary[j,ip1] == 5 or Surface_boundary[j,ip1] == 2) and (Surface_boundary[j,i-1] == 1 or Surface_boundary[j,i-1] == 0 or Surface_boundary[j,i-1] == 3) and (Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2) and (Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2):    
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

                    term3 = 0
                
                if (Surface_boundary[j,ip1] == 5 or Surface_boundary[j,ip1] == 2) and (Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2) and (Surface_boundary[j+1,i] == 1 or Surface_boundary[j+1,i] == 0 or Surface_boundary[j+1,i] == 3) and (Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2):
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j , i])
                    
                if (Surface_boundary[j,ip1] == 5 or Surface_boundary[j,ip1] == 2) and (Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2) and (Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2) and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 0 or Surface_boundary[j-1,i] == 3):
                     factor1 = csc2[j - 1] / (h ** 2)
                     term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

                     factor2 = 1 / (h ** 2)
                     term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

                     term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j , i] - temperature[j - 1, i])
                     
                if (Surface_boundary[j,ip1] == 1 or Surface_boundary[j,ip1] == 0 or Surface_boundary[j,ip1] == 3) and (Surface_boundary[j,i-1] == 1 or Surface_boundary[j,i-1] == 0 or Surface_boundary[j,i-1] == 3) and (Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2) and (Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2):
                    
                   factor1 = csc2[j - 1] / (h ** 2)
                   term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

                   factor2 = 1 / (h ** 2)
                   term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

                   term3 = 0

                if (Surface_boundary[j,ip1] == 1 or Surface_boundary[j,ip1] == 0 or Surface_boundary[j,ip1] == 3) and (Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2) and (Surface_boundary[j+1,i] == 1 or Surface_boundary[j+1,i] == 0 or Surface_boundary[j+1,i] == 3)  and (Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2):
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j , i])

                    
                if (Surface_boundary[j,ip1] == 1 or Surface_boundary[j,ip1] == 0 or Surface_boundary[j,ip1] == 3) and (Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2) and (Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2) and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 0 or Surface_boundary[j-1,i] == 3):
                   factor1 = csc2[j - 1] / (h ** 2)
                   term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

                   factor2 = 1 / (h ** 2)
                   term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

                   term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j , i] - temperature[j - 1, i])

                if (Surface_boundary[j,ip1] == 5 or Surface_boundary[j,ip1] == 2) and (Surface_boundary[j,i-1] == 1 or Surface_boundary[j,i-1] == 0 or Surface_boundary[j,i-1] == 3) and (Surface_boundary[j+1,i] == 1 or Surface_boundary[j+1,i] == 0 or Surface_boundary[j+1,i] == 3) and (Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2):
                   factor1 = csc2[j - 1] / (h ** 2)
                   term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

                   factor2 = 1 / (h ** 2)
                   term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

                   term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j, i])

                if (Surface_boundary[j,ip1] == 5 or Surface_boundary[j,ip1] == 2) and (Surface_boundary[j,i-1] == 1 or Surface_boundary[j,i-1] == 0 or Surface_boundary[j,i-1] == 3) and (Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2) and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 0 or Surface_boundary[j-1,i] == 3):
                   factor1 = csc2[j - 1] / (h ** 2)
                   term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

                   factor2 = 1 / (h ** 2)
                   term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j , i])

                   term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j, i] - temperature[j - 1, i])

     
                if (Surface_boundary[j,ip1] == 5 or Surface_boundary[j,ip1] == 2) and (Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2) and (Surface_boundary[j+1,i] == 1 or Surface_boundary[j+1,i] == 0 or Surface_boundary[j+1,i] == 3) and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 0 or Surface_boundary[j-1,i] == 3):
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])

                if (Surface_boundary[j,ip1] == 1 or Surface_boundary[j,ip1] == 0 or Surface_boundary[j,ip1] == 3) and (Surface_boundary[j,i-1] == 1 or Surface_boundary[j,i-1] == 0 or Surface_boundary[j,i-1] == 3) and (Surface_boundary[j+1,i] == 1 or Surface_boundary[j+1,i] == 0 or Surface_boundary[j+1,i] == 3) and (Surface_boundary[j-1,i] == 5 or Surface_boundary[j-1,i] == 2):
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j , i])

                                 
                if (Surface_boundary[j,ip1] == 5 or Surface_boundary[j,ip1] == 2) and (Surface_boundary[j,i-1] == 1  or Surface_boundary[j,i-1] == 0 or Surface_boundary[j,i-1] == 3) and (Surface_boundary[j+1,i] == 1 or Surface_boundary[j+1,i] == 0 or Surface_boundary[j+1,i] == 3) and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 0 or Surface_boundary[j-1,i] == 3):
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, i])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])

                if (Surface_boundary[j,ip1] == 1 or Surface_boundary[j,ip1] == 0 or Surface_boundary[j,ip1] == 3) and (Surface_boundary[j,i-1] == 5 or Surface_boundary[j,i-1] == 2) and (Surface_boundary[j+1,i] == 1 or Surface_boundary[j+1,i] == 0 or Surface_boundary[j+1,i] == 3)  and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 0 or Surface_boundary[j-1,i] == 3):
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i ] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])

                if (Surface_boundary[j,ip1] == 1 or Surface_boundary[j,ip1] == 0 or Surface_boundary[j,ip1] == 3) and (Surface_boundary[j,i-1] == 1 or Surface_boundary[j,i-1] == 0 or Surface_boundary[j,i-1] == 3) and (Surface_boundary[j+1,i] == 5 or Surface_boundary[j+1,i] == 2) and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 0 or Surface_boundary[j-1,i] == 3):
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j , i] - temperature[j - 1, i])

                    
                if (Surface_boundary[j,ip1] == 1 or Surface_boundary[j,ip1] == 0 or Surface_boundary[j,ip1] == 3) and (Surface_boundary[j,i-1] == 1 or Surface_boundary[j,i-1] == 0  or Surface_boundary[j,i-1] == 3) and (Surface_boundary[j+1,i] == 1 or Surface_boundary[j+1,i] == 0 or Surface_boundary[j+1,i] == 3) and (Surface_boundary[j-1,i] == 1 or Surface_boundary[j-1,i] == 0 or Surface_boundary[j-1,i] == 3): 
                    factor1 = csc2[j - 1] / (h ** 2)
                    term1 = factor1 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) * temperature[j, i - 1] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j, ip1] - diffusion_coeff[j, i - 1])) *  temperature[j, ip1])

                    factor2 = 1 / (h ** 2)
                    term2 = factor2 * (-2 * diffusion_coeff[j, i] * temperature[j, i] + (diffusion_coeff[j, i] - 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j - 1, i] + (diffusion_coeff[j, i] + 0.25 * (diffusion_coeff[j + 1, i] - diffusion_coeff[j - 1, i])) * temperature[j + 1, i])

                    term3 = cot[j - 1] * diffusion_coeff[j, i] * 0.5 / h * (temperature[j + 1, i] - temperature[j - 1, i])

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

def plot_annual_temperature(annual_temperature, average_temperature, title):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_temperature)
    plt.plot(average_temperature * np.ones(ntimesteps), label="average temperature")
    plt.plot(annual_temperature, label="annual temperature")

    plt.xlim((0, ntimesteps - 1))
    labels = [ "Nordpol" ,   "Nrd.Halbkugel",   "Äquator", "Sdl. Halbkugel" ,"Südpol" ]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("surface temperature [Â°C]")
    plt.grid()
    plt.title(title)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
    
def plot_inital_temperature(annual_temperature, average_temperature,title):   
    vmin = np.amin(annual_temperature)
    vmax = np.amax(annual_temperature)*1.044568245
    levels = np.linspace(vmin, vmax, 200)

    # Reuse plotting function from milestone 1.
    cbar = plot_robinson_projection(annual_temperature, title,levels=levels, cmap="Reds", vmin=vmin, vmax=vmax)
    cbar.set_label("temperature")

    # Adjust size of plot to viewport to prevent clipping of the legend.
    plt.tight_layout()
    plt.show()
    
    
def plot_annual_ice_thickness(annual_ice_thickness, average_ice_thickness, title):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_ice_thickness)
    plt.plot(average_ice_thickness * np.ones(ntimesteps), label="average ice thickness")
    plt.plot(annual_ice_thickness, label="annual ice thickness")

    plt.xlim((0, ntimesteps - 1))
    labels = [ "Nordpol" ,   "Nrd.Halbkugel",   "Äquator", "Sdl. Halbkugel" ,"Südpol" ]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("Hi")
    plt.grid()
    plt.title(title)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()    
    


def calc_mean(data, area):
    nlatitude, nlongitude = data.shape

    mean_data = area[0] * data[0, 0] + area[-1] * data[-1, -1]
    for i in range(1, nlatitude - 1):
        for j in range(nlongitude):
            mean_data += area[i] * data[i, j]

    return mean_data


def calc_mean_ocn_south(data, area):
    nlatitude, nlongitude = data.shape
    mean_data = 0
    j_equator = int((nlatitude - 1) / 2)
    area_normalized = 0
    # South Pole
    if np.isnan(data[-1, -1]) == False:
       mean_data = area[-1] * data[-1, -1]
       area_normalized += area[-1] 
    else:
        mean_data = 0
    # Inner nodes
    for j in range(j_equator+1, nlatitude-1):
        for i in range(nlongitude):
            if np.isnan(data[j, i]) == False:
               mean_data += area[j] * data[j, i]         
               area_normalized += area[j]
    # Equator
    for i in range(nlongitude):
        if np.isnan(data[j_equator, i]) == False:
           mean_data += 0.5 * area[j_equator] * data[j_equator, i]
           area_normalized += 0.5* area[j_equator]
    if  area_normalized !=0:     
        return  mean_data *  1/area_normalized
    else: 
        return 0

def calc_mean_ocn_north(data, area):
    nlatitude, nlongitude = data.shape
    mean_data = 0
    j_equator = int((nlatitude - 1) / 2)
    area_normalized = 0
    # North Pole
    if np.isnan(data[0, 0]) == False:
       mean_data = area[0] * data[0, 0]
       area_normalized += area[0] 
    else:
        mean_data = 0
    # Inner nodes

    for j in range(1, j_equator):
        for i in range(nlongitude):
            if np.isnan(data[j, i]) == False:
               mean_data += area[j] * data[j, i]      
               area_normalized += area[j]
    # Equator
    for i in range(nlongitude):
        if np.isnan(data[j_equator, i]) == False:
           mean_data += 0.5 * area[j_equator] * data[j_equator, i]
           area_normalized += 0.5* area[j_equator]
          
    if  area_normalized !=0:     
        return  mean_data *  1/area_normalized
    else: 
        return 0
 

def calc_mean_ocn(data, area):
    nlatitude, nlongitude = data.shape
    mean_data = 0
    area_normalized = 0
    # North Pole
    if np.isnan(data[0, 0]) == False:
       mean_data = area[0] * data[0, 0]
       area_normalized += area[0] 
    # South Pole
    if np.isnan(data[-1, -1]) == False:
       mean_data += area[-1] * data[-1, -1]
       area_normalized += area[-1] 
       
    for j in range(1,nlatitude-1):
        for i in range(nlongitude):
            if (np.isnan(data[j,i]) == False):
                mean_data += area[j] * data[j,i]  
                area_normalized += area[j] 
   
    return mean_data * 1/area_normalized


def calc_mean_north(data, area):
    nlatitude, nlongitude = data.shape
    j_equator = int((nlatitude - 1) / 2)

    # North Pole
    mean_data = area[0] * data[0, 0]

    # Inner nodes
    for j in range(1, j_equator):
        for i in range(nlongitude):
            mean_data += area[j] * data[j, i]

    # Equator
    for i in range(nlongitude):
        mean_data += 0.5 * area[j_equator] * data[j_equator, i]

    return 2 * mean_data


def calc_mean_south(data, area):
    nlatitude, nlongitude = data.shape
    j_equator = int((nlatitude - 1) / 2)

    # South Pole
    mean_data = area[-1] * data[-1, -1]

    # Inner nodes
    for j in range(j_equator + 1, nlatitude - 1):
        for i in range(nlongitude):
            mean_data += area[j] * data[j, i]

    # Equator
    for i in range(nlongitude):
        mean_data += 0.5 * area[j_equator] * data[j_equator, i]

    return 2 * mean_data


def calc_area(nlatitude, nlongitude):
    area = np.zeros(nlatitude, dtype=np.float64)
    delta_theta = np.pi / (nlatitude - 1)

    # Poles
    area[0] = area[-1] = 0.5 * (1 - np.cos(0.5 * delta_theta))

    # Inner cells
    for j in range(1, nlatitude - 1):
        area[j] = np.sin(0.5 * delta_theta) * np.sin(delta_theta * j) / nlongitude

    return area


def calc_lambda(dt = 1.0 / 48,  nt=48, ecc=  0.016740, per = 1.783037):  #calculation of true longitude
    eccfac = 1.0 - ecc**2
    rzero  = (2.0*np.pi)/eccfac**1.5
  
    lambda_ = np.zeros(nt)
    
    for n in range(1, nt): 
    
      nu = lambda_[n-1] - per
      t1 = dt*(rzero*(1.0 - ecc * np.cos(nu))**2)
      t2 = dt*(rzero*(1.0 - ecc * np.cos(nu+0.5*t1))**2)
      t3 = dt*(rzero*(1.0 - ecc * np.cos(nu+0.5*t2))**2)
      t4 = dt*(rzero*(1.0 - ecc * np.cos(nu + t3))**2)
      lambda_[n] = lambda_[n-1] + (t1 + 2.0*t2 + 2.0*t3 + t4)/6.0

    return lambda_

def plot_annual_T_S_vgl(annual_temperature_og,  annual_temperature_paper):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_temperature_og)
    plt.plot(annual_temperature_og, label="temperature (EBM)")
    plt.plot(annual_temperature_paper, label="temperature paper")
   

    plt.xlim((0, ntimesteps - 1))
    #labels = ["Äquator" ,"Nrd. Wendekreis", "Nrd. Polarkreis", "Nordpol"  ]
    labels = ["0°", "18°", "36°" , "54°", "72°", "90°" ]
    plt.xticks(np.linspace(0, ntimesteps - 1, 6), labels)
    ax.set_ylabel("surface temperature [Â°C]")
    plt.grid()
    plt.title("Comparison Surface temperature ")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
    
def plot_annual_ice_thickness_vgl(annual_ice_thickness,  annual_ice_thickness_paper, titel):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_ice_thickness)
    plt.plot(annual_ice_thickness, label="Ice thickness (EBM)")
    plt.plot(annual_ice_thickness_paper, label="Ice thickness (paper)")
   

    plt.xlim((0, ntimesteps - 1))
    labels = ["March", "June", "September", "December", "March"]
    #labels = ["Südpol", "Sdl. Halbkugel",   "Äquator" ,   "Nrd.Halbkugel", "Nordpol"  ]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("H_i")
    plt.grid()
    plt.title(titel)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()    
    
def plot_annual_ice_edge_vgl(annual_ice_edge, annual_ice_edge_paper):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_ice_edge)
    plt.plot(annual_ice_edge, label="Ice-edge (EBM)")
    plt.plot(annual_ice_edge_paper, label="Ice-edge (paper)")
   

    plt.xlim((0, ntimesteps - 1))
    labels = ["March", "June", "September", "December", "March"]
    #labels = ["Südpol", "Sdl. Halbkugel",   "Äquator" ,   "Nrd.Halbkugel", "Nordpol"  ]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("$\phi$i")
    #plt.rc('axes', labelsize = 20) #geht nicht für nur einen Plot?
    plt.grid()
    plt.title("Comparison Ice-edge")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()     
    
    
    
def plot_temperature_time(annual_temperature_total, average_temperature , titel):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_temperature_total)
    plt.plot(annual_temperature_total, label="temperature (total)")
    plt.plot(average_temperature * np.ones(ntimesteps), label="average temperature")

    plt.xlim((0, ntimesteps - 1))
    labels = ["March", "June", "September", "December", "March"]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("surface temperature [Â°C]")
    plt.grid()
    plt.title(titel)
    plt.legend(loc="upper right")
    

    plt.tight_layout()
    plt.show()    
    
def plot_ice_edge_time(annual_ice_edge_total, average_ice_edge , titel):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_ice_edge_total)
    plt.plot(annual_ice_edge_total, label="Ice-edge")
    plt.plot(average_ice_edge  * np.ones(ntimesteps), label="average Ice-edge")

    plt.xlim((0, ntimesteps - 1))
    labels = ["March", "June", "September", "December", "March"]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("$\phi$i (°N)")
    #plt.rc('axes', labelsize = 20)
    plt.grid()
    plt.title(titel)
    plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()        
    
def plot_ice_thickness_time(annual_ice_thickness_total, average_ice_thickness, titel):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_ice_thickness_total)
    plt.plot(annual_ice_thickness_total, label="ice thickness")
    plt.plot(average_ice_thickness * np.ones(ntimesteps), label="average ice thickness")

    plt.xlim((0, ntimesteps - 1))
    labels = ["March", "June", "September", "December", "March"]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("Hi")
    plt.grid()
    plt.title(titel)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()        
    
def plot_temperature_latitude(annual_temperature_total, average_temperature , titel):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_temperature_total)
    plt.plot(annual_temperature_total, label="temperature (total)")
    plt.plot(average_temperature * np.ones(ntimesteps), label="average temperature")
   

    plt.xlim((0, ntimesteps - 1))
    labels = [ "Nordpol" ,   "Nrd.Halbkugel",   "Äquator", "Sdl. Halbkugel" ,"Südpol" ]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("surface temperature [Â°C]")
    plt.grid()
    plt.title(titel)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()        


def plot_annual_temperature_north_south(annual_temperature_north, annual_temperature_south, annual_temperature_total,
                                        average_temperature_north, average_temperature_south,
                                        average_temperature_total, titel):
    fig, ax = plt.subplots()

    ntimesteps = len(annual_temperature_total)
    plt.plot(average_temperature_total * np.ones(ntimesteps), label="average temperature (total)")
    plt.plot(average_temperature_north * np.ones(ntimesteps), label="average temperature (north)")
    plt.plot(average_temperature_south * np.ones(ntimesteps), label="average temperature (south)")
    plt.plot(annual_temperature_total, label="temperature (total)")
    plt.plot(annual_temperature_north, label="temperature (north)")
    plt.plot(annual_temperature_south, label="temperature (south)")

    plt.xlim((0, ntimesteps - 1))
    labels = ["March", "June", "September", "December", "March"]
    plt.xticks(np.linspace(0, ntimesteps - 1, 5), labels)
    ax.set_ylabel("surface temperature [°C]")
    plt.grid()
    plt.title(titel)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()