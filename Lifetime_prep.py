#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:58:59 2023

@author: cifuente
"""

'''
Interpolate to 13:30 LT, which is 11:30 in the model for summer time

'''

import shutil
import os
import datetime
import xarray as xr
from netCDF4 import Dataset
import numpy as np
import scipy.constants as sc

begin_time = datetime.datetime.now()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#Read the needed data

# OH originaly in mol/mol (Convert to molecule/cm3)
# T in K
# P in Pa
# sc.R m3 pa K-1 mol-1

lotos_dir = '/net/pc190622/nobackup/users/cifuente/TROPOMIvsMODELS_DATA/LOTOS_EUROS/SUMMER/'
lotos_files = sorted(os.listdir(lotos_dir))

for i in lotos_files:
    
    print(i)
    
    lotos = lotos_dir + '/' + i
    date = i[21:29]
    
    DS = xr.open_dataset(lotos)

    #Extract data needed
    time = DS.time.values
    lat = DS.latitude.values
    lon = DS.longitude.values
    oh = DS.oh.values
    T = DS.T.values
    P_hl = DS.hp.values
    
    a,b,c,d = oh.shape
    
    P_fl = np.empty((a, b, c, d))
    
    for ilev in np.arange(b):
        P_fl[:,ilev,:,:] = (P_hl[:,ilev+1,:,:] + P_hl[:,ilev,:,:])/2


    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Unit conversion and lifetime at different vertical levels
    
    #Unit conversion for oh mol/mol -> molecule/cm3
    oh_moleccm3 = ( oh *  P_fl /(sc.R * T) * sc.Avogadro ) / 1000000
    
    
    print(np.nanmean(oh_moleccm3[:,0,:,:]))
    
    #Estimate lifetime
    K = 2.8e-11*((T/300)**(-1.3))
    Tau_h = (1 / (K * oh_moleccm3)) / 3600
    
    
    print([np.percentile(Tau_h[11,:,:,:],1), np.percentile(Tau_h[11,:,:,:],99)])
    print(time)
    
    #time, level, latitude, longitude
    savefile = '/net/pc190622/nobackup/users/cifuente/TROPOMIvsMODELS_DATA/LOTOS_LIFETIME/Lifetime_XXX' + date + '.nc'
    s = xr.Dataset(
        data_vars=dict(
            tau=(["time","lev", "XLAT", "XLONG"], Tau_h),
            ),
        coords=dict(
            time=(["time"], time),
            level=(["lev"], np.arange(11)),
            latitude=(["XLAT"], lat),
            longitude=(["XLONG"], lon),
            ),
        )
    s.to_netcdf(savefile)