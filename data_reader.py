#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 12, 2023

@author: Felipe Cifuentes
"""

import xarray as xr
from netCDF4 import Dataset
import numpy as np


def data_lotos3d(path_lotos3d, time_in):
            
    lotos3d = xr.open_dataset(path_lotos3d)
    
    no2_mol_mol = lotos3d.no2.interp(time=time_in)
    no2_mol_mol = no2_mol_mol.transpose("level", "latitude", "longitude")
    
    no_mol_mol = lotos3d.no.interp(time=time_in)
    no_mol_mol = no_mol_mol.transpose("level", "latitude", "longitude")
    
    
    p_hl = lotos3d.hp.interp(time=time_in)
    z_hl = lotos3d.halt.interp(time=time_in)
    
    T_lotos = lotos3d.T.interp(time=time_in)
    
    
    # Derive full-levels pressure and altiudes
    nlev, nlat, nlon = no2_mol_mol.shape
    p_fl = np.empty((nlev, nlat, nlon))
    z_fl = np.empty((nlev, nlat, nlon))

    for ilev in np.arange(nlev):
        # full levels
        p_fl[ilev,:,:] = (p_hl[ilev+1,:,:] + p_hl[ilev,:,:])/2
        z_fl[ilev,:,:] = (z_hl[ilev+1,:,:] + z_hl[ilev,:,:])/2
        
    return no_mol_mol, no2_mol_mol, p_hl, z_hl, p_fl, z_fl, T_lotos


def data_lotos_emis(path_emis_bio, path_emis_ant, time_in):
    
    #Final emiss unit are mol/m2-s
    
    bio = xr.open_dataset(path_emis_bio)
    ant = xr.open_dataset(path_emis_ant)
    
    no_bio = bio.no.interp(time=time_in)
    no_ant = ant.no.interp(time=time_in)
    no2_ant = ant.no2.interp(time=time_in)
    
    nox_emis = no_bio * (1000/30) + no_ant * (1000/30) + no2_ant * (1000/46)
    nox_emis = nox_emis.transpose("level","latitude", "longitude")
    
    return nox_emis

def data_lotos_wind(path_wind, time_in):
        
    wind = xr.open_dataset(path_wind)
    
    U = wind.uv_u.interp(time=time_in)
    V = wind.uv_v.interp(time=time_in)
    
    return U, V

def data_lotos_pblh(path_pblh, time_in):

    pbl = xr.open_dataset(path_pblh)
    pbl = pbl.blh.interp(time=time_in).data
    
    return pbl