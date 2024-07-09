#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 12, 2023

@author: Felipe Cifuentes
"""

#%% Import functions
import os
import datetime
import xarray as xr
import numpy as np
import datetime

#%% Import costum modules
from data_reader import *
from auxiliary import *


#%% Define input and output paths
lotos_conce = '/net/pc190622/nobackup/users/cifuente/TROPOMIvsMODELS_DATA/LOTOS_EUROS/SUMMER/'
lotos_emiss = '/net/pc190622/nobackup/users/cifuente/TROPOMIvsMODELS_DATA/LOTOS_EUROS_EMIS/SUMMER/'
lotos_meteo = '/net/pc190622/nobackup/users/cifuente/TROPOMIvsMODELS_DATA/LOTOS_EUROS_METEO/SUMMER/'

lotos_tau = '/net/pc190622/nobackup/users/cifuente/TROPOMIvsMODELS_DATA/LOTOS_LIFETIME/' # Files need to be previously generated using de lifetime_prep.py code

output_nc = '/net/pc190622/nobackup/users/cifuente/FDA_OUT/TEST_PAPER.nc'
output_png = '/net/pc190622/nobackup/users/cifuente/FDA_OUT/TEST_PAPER.png'

#%% User defined options
hour_utc = '11:30'      # Hour in UTC when emissions will be evaluated
colu_opt = 2            # Altitude of the NO2 column. (1) Full troposphere (2) Up to PBL height
nume_opt = 2            # Numerical method for resolving the divergence term. (1) 2nd order (2) 4th order
dive_opt = 1            # Expresion used to evaluate the divergence term. (1) w ∇(LV) + LV ∇(w) (2) w ∇(LV)
wdiv_opt = 1            # Defines if the wind divergence is mitigated. (1) Use wind as supply by the model (2) Remove divergence from wind fields
wind_opt = 2            # Altitude for extracting the wind fields. (1) Fixed altitud strategy (2) Function of the PBL
life_opt = 2            # Approach for deriving the NO2 lifetime. (1) Fixed single value (2) Kinetic equation with modeling outputs
upsc_opt = 2            # Approach for deriving the NO2 to NOx factor. (1) Fixed single value (2) Modeling outputs


wind_alt = 500          # Fixed altitud in m for selecting the wind fields. Only used if opt 1 was selected for wind_opt
wind_pbl = 0.5          # Fration of the PBL at which the wind fields will be extracted. Only used if opt 2 was selected for wind_opt
life_tau = 4            # NO2 lifetime in hours. Only used if opt 1 was selected for life_opt
upsc_fac = 1.32         # NO2 to NOX scaling factor. Only used if opt 1 was used for upsc_opt

#%% Code initialization
begin_time = datetime.datetime.now()
lotos_file = sorted(os.listdir(lotos_conce))

lat = xr.open_dataset(lotos_conce + lotos_file[0]).latitude.values      # Extract latitudes
lon = xr.open_dataset(lotos_conce + lotos_file[0]).longitude.values     # Extract longitudes

dlat = lat[1] - lat[0]                                                  # Latitude spacing in degrees
dlon = lon[1] - lon[0]                                                  # Longitude spacing in degrees 

dlat_m = dlat * 110574                                                  # Latitude spacing in meters
dlon_m = dlon * 111320 * np.cos(lat[0] * np.pi / 180)                   # Longitude spacing in meters

nfil = len(lotos_file)
nlat = len(lat)
nlon = len(lon)

# Create matrixes that will be populated later
final_u = np.zeros((nfil,nlat,nlon))*np.nan
final_v = np.zeros((nfil,nlat,nlon))*np.nan
final_div = np.zeros((nfil,nlat,nlon))*np.nan
final_no2 = np.zeros((nfil,nlat,nlon))*np.nan
final_nox = np.zeros((nfil,nlat,nlon))*np.nan
final_tau = np.zeros((nfil,nlat,nlon))*np.nan
final_tem = np.zeros((nfil,nlat,nlon))*np.nan
final_emi_fda = np.zeros((nfil,nlat,nlon))*np.nan
final_emi_mod = np.zeros((nfil,nlat,nlon))*np.nan

#%% Code execution
z = 0
for i in lotos_file:
    
    lotos = lotos_conce + i
    date = i[21:29]
    time = np.datetime64(date[0:4] + '-' + date[4:6] + '-' + date[6:8] + 'T' + hour_utc + ':00.000000000')
    
    print('Processing file : ' + str(time))
    
    # Define files path and names for extracting the data
    emis_bio = lotos_emiss + 'LEKF_NKS-pre_emis_bio_' + date + '.nc'
    emis_ant = lotos_emiss + 'LEKF_NKS-pre_emis_' + date + '.nc'
    wind = lotos_meteo + 'LEKF_NKS-pre_meteo-3d_' + date + '.nc'
    pblh = lotos_meteo + 'LEKF_NKS-pre_meteo_' + date + '.nc'
    
    # Read data
    no_mol_mol, no2_mol_mol, p_hl, z_hl, p_fl, z_fl, t = data_lotos3d(lotos, time)
    nox_emis = data_lotos_emis(emis_bio, emis_ant, time)
    u, v = data_lotos_wind(wind, time)
    pbl = data_lotos_pblh(pblh, time)
    
    #%% Estimate NO2 column
    if colu_opt == 1:
        
        no2_colum, no2_layer = column_troposphere(no2_mol_mol, p_hl, 'NO2')
        no_colum, no_layer = column_troposphere(no_mol_mol, p_hl, 'NO')
    
    else:
        
        no2_colum, no2_layer = column_pbl(no2_mol_mol, p_hl, 'NO2', z_fl, pbl)
        no_colum, no_layer = column_pbl(no_mol_mol, p_hl, 'NO', z_fl, pbl)

    #%% Escale NO2 to NOx
    if upsc_opt == 1:
        nox_colum = 1.32 * no2_colum
        
    else:
        nox_colum = no2_colum + no_colum

    #%% Extract wind information
    if wind_opt == 1:
        alt = np.ones((nlat,nlon)) * wind_alt
    
    else:
        alt = pbl * wind_pbl
    
    u_interp, v_interp = wind_altitud(u.data, v.data, z_fl, alt)
    
    #%% Mitigate wind divergence when necesarry
    if wdiv_opt == 2:
        u_interp, v_interp = reduce_wind_div(u_interp,v_interp,dlat_m,dlon_m)
    
    #%% Estimate the flux divergence
    if dive_opt == 1:
        
        # Terms for the expanded divergence
        LV = nox_colum
        w_u = u_interp
        w_v = v_interp
        
        # Compute divergence    
        divergence = np.zeros((nlat,nlon))*np.nan # Create a matrix full of nan to populate later
        
        if nume_opt == 1:
            
            for n in np.arange(1,nlat-1,1):
                for m in np.arange(1,nlon-1,1):
                    divergence[n,m] = w_v[n,m]*((LV[n+1,m]-LV[n-1,m])/(2*dlat_m)) + w_u[n,m]*((LV[n,m+1]-LV[n,m-1])/(2*dlon_m)) + LV[n,m]*((w_v[n+1,m]-w_v[n-1,m])/(2*dlat_m)) + LV[n,m]*((w_u[n,m+1]-w_u[n,m-1])/(2*dlon_m))
                    
        else:
            
            for n in np.arange(2,nlat-2,1):
                for m in np.arange(2,nlon-2,1):
                    divergence[n,m] = w_v[n,m]*((LV[n-2,m]-8*LV[n-1,m]+8*LV[n+1,m]-LV[n+2,m])/(12*dlat_m)) + w_u[n,m]*((LV[n,m-2]-8*LV[n,m-1]+8*LV[n,m+1]-LV[n,m+2])/(12*dlon_m)) + LV[n,m]*((w_v[n-2,m]-8*w_v[n-1,m]+8*w_v[n+1,m]-w_v[n+2,m])/(12*dlat_m)) + LV[n,m]*((w_u[n,m-2]-8*w_u[n,m-1]+8*w_u[n,m+1]-w_u[n,m+2])/(12*dlon_m))
    
    else:
        
        # Terms for the expanded divergence
        LV = nox_colum
        w_u = u_interp
        w_v = v_interp
        
        # Compute divergence    
        divergence = np.zeros((nlat,nlon))*np.nan # Create a matrix full of nan to populate later
        
        if nume_opt == 1:
            
            for n in np.arange(1,nlat-1,1):
                for m in np.arange(1,nlon-1,1):
                    divergence[n,m] = w_v[n,m]*((LV[n+1,m]-LV[n-1,m])/(2*dlat_m)) + w_u[n,m]*((LV[n,m+1]-LV[n,m-1])/(2*dlon_m)) 
                    
        else:
            
            for n in np.arange(2,nlat-2,1):
                for m in np.arange(2,nlon-2,1):
                    divergence[n,m] = w_v[n,m]*((LV[n-2,m]-8*LV[n-1,m]+8*LV[n+1,m]-LV[n+2,m])/(12*dlat_m)) + w_u[n,m]*((LV[n,m-2]-8*LV[n,m-1]+8*LV[n,m+1]-LV[n,m+2])/(12*dlon_m))
    
    #%% Adding the sink term

    # Convert to emissiones
    if life_opt ==1 :
        tau_column = life_tau * np.ones((nlat,nlon))
        emis = (nox_colum)/(tau_column*3600) + divergence

    else:
        
        lifetime_path =  lotos_tau + 'Lifetime_' + date + '.nc'
        tau_column = no2_lifetime(lifetime_path, no2_layer, pbl, z_fl, hour_utc)
        emis = (nox_colum)/(tau_column*3600) + divergence


    #%% Store data of each iteration
    final_u[z,:,:] = u_interp
    final_v[z,:,:] = v_interp
    final_div[z,:,:] = divergence
    final_no2[z,:,:] = no2_colum
    final_nox[z,:,:] = nox_colum
    final_tau[z,:,:] = tau_column
    final_tem[z,:,:] = t.values[0,:,:]
    final_emi_fda[z,:,:] = emis
    final_emi_mod[z,:,:] = nox_emis.values

    
    z = z +1
    
#%% Export netCDF outputs    
# date = []
# for i in lotos_file:
#     date.append(i[21:29])
    
# s = xr.Dataset(
#     data_vars=dict(
#         DIV=(["date", "XLAT", "XLONG"], final_div),
#         U=(["date", "XLAT", "XLONG"], final_u),
#         V=(["date", "XLAT", "XLONG"], final_v),
#         NO2_COL=(["date", "XLAT", "XLONG"], final_no2),
#         NOX_COL=(["date", "XLAT", "XLONG"], final_nox),
#         NOX_FDA_EMIS=(["date", "XLAT", "XLONG"], final_emi_fda),
#         NOX_MOD_EMIS=(["date", "XLAT", "XLONG"], final_emi_mod),
#         lifetime =(["date", "XLAT", "XLONG"], final_tau), 
#         temp =(["date", "XLAT", "XLONG"], final_tem),
#         ),
#     coords=dict(
#         time=(["date"], date),
#         lon=(["XLONG"], lon),
#         lat=(["XLAT"], lat),
#         ),
#     )
# s.to_netcdf(output_nc)

s = xr.Dataset(
    data_vars=dict(
        DIV=(["XLAT", "XLONG"], np.nanmean(final_div, axis=0)),
        U=(["XLAT", "XLONG"], np.nanmean(final_u, axis=0)),
        V=(["XLAT", "XLONG"], np.nanmean(final_v, axis=0)),
        NO2_COL=(["XLAT", "XLONG"], np.nanmean(final_no2, axis=0)),
        NOX_COL=(["XLAT", "XLONG"], np.nanmean(final_nox, axis=0)),
        NOX_FDA_EMIS=(["XLAT", "XLONG"], np.nanmean(final_emi_fda, axis=0)),
        NOX_MOD_EMIS=(["XLAT", "XLONG"], np.nanmean(final_emi_mod, axis=0)),
        lifetime =(["XLAT", "XLONG"], np.nanmean(final_tau, axis=0)), 
        ),
    coords=dict(
        lon=(["XLONG"], lon),
        lat=(["XLAT"], lat),
        ),
    )
s.to_netcdf(output_nc)

#%% Create average plot and compute performance metrics
fda_plot(lat, lon, final_emi_fda, final_emi_mod, final_div, output_png)
fda_metrics(np.nanmean(final_emi_fda, axis=0), np.nanmean(final_emi_mod, axis=0))

#-------------------------------------------------------------------------------------------------------------------------------------------------------
print('**********************************************************************')
print('Done ! - Execution time:')
print(datetime.datetime.now() - begin_time)
print('**********************************************************************')


