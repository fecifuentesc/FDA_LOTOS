#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 12, 2023

@author: Felipe Cifuentes
"""


import numpy as np
import xarray as xr

#%%
def column_troposphere(no2_lotos, p_hl_lotos, pollutant):
    
    # Mixing ratio and column diference
    
    if pollutant == 'NO':
        no2mass = 30   # g/mol
        
    if pollutant == 'NO2':
        no2mass = 46   # g/mol
    
    g = 9.80665    #m/s2
    Nav = 6.022137e23
    xmair = 28.94
    pdiff2moleccm2 = 1.0e-15 * 0.1 * Nav / ( g * xmair )  #Pmolec de aire por cm2 
    
    # Initialize a matrix to store the results
    a, b, c = no2_lotos.shape
    NO2_layer = np.zeros((a, b, c))

    for n in range(a):
        NO2_layer[n,:,:] = pdiff2moleccm2 * (p_hl_lotos[n,:,:] - p_hl_lotos[n+1,:,:]) * no2_lotos[n,:,:]

    NO2_colum = np.sum(NO2_layer,axis=0)
    
    #Change to mol/m2
    NO2_layer = NO2_layer / (6.022141e+19/1e15)
    NO2_colum = NO2_colum / (6.022141e+19/1e15)
    
    return NO2_colum, NO2_layer


#%%
def column_pbl(no2_lotos, p_hl_lotos, pollutant, z_fl_lotos, pbl):
    
    # Mixing ratio and column diference
    
    if pollutant == 'NO':
        no2mass = 30   # g/mol
        
    if pollutant == 'NO2':
        no2mass = 46   # g/mol
    
    g = 9.80665    #m/s2
    Nav = 6.022137e23
    xmair = 28.94
    pdiff2moleccm2 = 1.0e-15 * 0.1 * Nav / ( g * xmair )  #Pmolec de aire por cm2 
    
    # Initialize a matrix to store the results
    a, b, c = no2_lotos.shape
    NO2_layer = np.zeros((a, b, c))

    for n in range(a):
        NO2_layer[n,:,:] = pdiff2moleccm2 * (p_hl_lotos[n,:,:] - p_hl_lotos[n+1,:,:]) * no2_lotos[n,:,:]
    
    
    NO2_colum = np.empty((b, c))
    for x in range(b):
        for y in range(c):

            dif_alt = np.absolute(z_fl_lotos[:,x,y]-pbl[x,y])
            index = dif_alt.argmin() 
            
            if index == 0 :
                index = 1
    
            NO2_colum[x,y] =  np.sum(NO2_layer[0:index,x,y],axis=0)
    
    #Change to mol/m2
    NO2_layer = NO2_layer / (6.022141e+19/1e15)
    NO2_colum = NO2_colum / (6.022141e+19/1e15)
    
    return NO2_colum, NO2_layer


#%%
def wind_altitud(U, V, z_fl, altitude):
    
    a, b, c = U.shape
    U_interp = np.zeros((b, c))
    V_interp = np.zeros((b, c))
    
    # For linear interpolation
    for y in range(b):
        for z in range(c):
            
            dif_alt = np.absolute(z_fl[:,y,z]-altitude[y,z])
            index_alt1 = dif_alt.argmin()
            
            dif_alt[index_alt1] = dif_alt[index_alt1]*1e9
            index_alt2 = dif_alt.argmin()
            
            # Linear interpolation of U
            m = (U[index_alt1,y,z] - U[index_alt2,y,z]) / (z_fl[index_alt1,y,z] - z_fl[index_alt2,y,z])
            intercept = U[index_alt1,y,z] - m * z_fl[index_alt1,y,z]
            U_interp[y,z] = m * altitude[y,z] + intercept
            
            # Linear interpolation of V
            m = (V[index_alt1,y,z] - V[index_alt2,y,z]) / (z_fl[index_alt1,y,z] - z_fl[index_alt2,y,z])
            intercept = V[index_alt1,y,z] - m * z_fl[index_alt1,y,z]
            V_interp[y,z] = m * altitude[y,z] + intercept

    return U_interp, V_interp


#%%
def reduce_wind_div(U,V,dlat_m,dlon_m):
    
    nlat, nlon = U.shape
    
    Div = np.zeros((nlat,nlon))*np.nan # Create a matrix full of nan to populate later
    Div_new = np.zeros((nlat,nlon))*np.nan # Create a matrix full of nan to populate later
    Div_new_U = np.zeros((nlat,nlon))*np.nan # Create a matrix full of nan to populate later
    Div_new_V = np.zeros((nlat,nlon))*np.nan # Create a matrix full of nan to populate later
    Grad_U = np.zeros((nlat,nlon)) # Create a matrix full of nan to populate later
    Grad_V = np.zeros((nlat,nlon)) # Create a matrix full of nan to populate later
    
    #------------------------------------------------------------------------------------------------------------------------------
    # Compute initial divergence for comparison later
    for n in np.arange(1,nlat-1,1):
        for m in np.arange(1,nlon-1,1):
            Div[n,m] = (V[n+1,m]-V[n-1,m])/(2*dlat_m) + (U[n,m+1]-U[n,m-1])/(2*dlon_m)
    
    #------------------------------------------------------------------------------------------------------------------------------
    # Iterations to reduce the divergence
    U_new = np.copy(U)
    V_new = np.copy(V)
    
    sf = 0.2 # User define scaling factor
    cost_funct_old = 1e19
    Div_old = Div * 1e19
    
    print('Reducing wind divergence...')
    
    for j in np.arange(10000) :

        # Compute divergence of W
        for n in np.arange(1,nlat-1,1):
            Div_new_V[n,1:(nlon-1)] = (V_new[n+1,1:(nlon-1)]-V_new[n-1,1:(nlon-1)])/(2*dlat_m)
        for m in np.arange(1,nlon-1,1):
            Div_new_U[1:(nlat-1),m] = (U_new[1:(nlat-1),m+1]-U_new[1:(nlat-1),m-1])/(2*dlon_m)
        Div_new = Div_new_U + Div_new_V
        
        # Compute gradiant of the divergence
        for n in np.arange(2,nlat-2,1):
            Grad_V[n,2:(nlon-2)] = (Div_new[n+1,2:(nlon-2)]-Div_new[n-1,2:(nlon-2)])/(2*dlat_m)
        for m in np.arange(2,nlon-2,1):
            Grad_U[2:(nlat-2),m] = (Div_new[2:(nlat-2),m+1]-Div_new[2:(nlat-2),m-1])/(2*dlon_m)
        
        U_prev = np.copy(U_new)
        V_prev = np.copy(V_new)
                
        U_new = U_new + (dlon_m**2 * sf * Grad_U)
        V_new = V_new + (dlat_m**2 * sf * Grad_V)



        cost_func_max = np.absolute(np.nanmax(Div_new) - np.nanmax(Div_old))*1e4
        cost_func_min = np.absolute(np.nanmin(Div_new) - np.nanmin(Div_old))*1e4

        if cost_func_max < 1e-4 and cost_func_min < 1e-4:
            break
        
        #print([j, np.nanmax(Div_new)*1e4, np.nanmin(Div_new)*1e4, np.nanmean(Div_new)*1e4, cost_func_max, cost_func_min])

        #cost_funct_old = cost_funct_new
        Div_old = Div_new
    
    return U_new, V_new

#%%
def no2_lifetime(lifetime_path, no2_layer, pbl, z_fl, hour):
    
    lifetime = xr.open_dataset(lifetime_path)
    lifetime = lifetime.tau.data
    
    time = int(hour[0:2])
    lifetime = np.mean(lifetime[time-1:time,:,:],axis=0)
    
    nlev, nlat, nlon = lifetime.shape
    
    lifetime_column = np.zeros((nlat,nlon))*np.nan
    for x in range(nlat):
        
        for y in range(nlon):
            
            dif_alt = np.absolute(z_fl[:,x,y]-pbl[x,y])
            index = dif_alt.argmin()
            
            if index == 0 :
                index = 1
    
            no2_frac = np.zeros((index))*np.nan
            for z in range(index):
                no2_frac[z] = no2_layer[z,x,y] / np.sum(no2_layer[0:index,x,y],axis=0)
    
            # lifetime_weighted = lifetime[0:index,x,y] * no2_frac
            # lifetime_column[x,y] = np.sum(lifetime_weighted)
            
            #This is for 1/tau test.
            lifetime_weighted = (1 / lifetime[0:index,x,y]) * no2_frac
            lifetime_column[x,y] = 1 / np.sum(lifetime_weighted)
    
    return lifetime_column


#%%
def fda_plot(lat, lon, emis_fda, emis_mod, div, save_path):
    
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    import matplotlib.colors as colors
    from matplotlib.colors import LinearSegmentedColormap
    
    import cartopy.crs as ccrs
    from cartopy.feature import ShapelyFeature
    from cartopy.io.shapereader import Reader
    
    from scipy.signal import convolve2d
    
    shp_file = '/usr/people/cifuente/Downloads/Europe/Europe.shp'
    
    #----------------------------------------------------------------------------------------------------------------
    emis_fda = np.nanmean(emis_fda, axis=0) * 1e8
    emis_mod = np.nanmean(emis_mod, axis=0) * 1e8
    div = np.nanmean(div, axis=0) * 1e8
    
    
    psf = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
    emis_mod = convolve2d(emis_mod, psf, 'same')

    dif_emis = emis_fda - emis_mod

    colors = ["indigo", "blueviolet", "dodgerblue", "cyan", "white", "yellow", "orange", "red", "darkred"]
    colormap = LinearSegmentedColormap.from_list("mycmap", colors)
    colormap2 = get_cmap('bwr')

    ## Elaboracion del grafico
    fig, axs = plt.subplots(2, 2, figsize=(9,8.5),subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

    c0=axs[0,0].pcolormesh(lon, lat, emis_mod, vmin=-8, vmax=8, transform=ccrs.PlateCarree(), cmap=colormap)
    c1=axs[0,1].pcolormesh(lon, lat, emis_fda, vmin=-8, vmax=8, transform=ccrs.PlateCarree(), cmap=colormap)
    c2=axs[1,0].pcolormesh(lon, lat, div, vmin=-8, vmax=8, transform=ccrs.PlateCarree(), cmap=colormap)
    c3=axs[1,1].pcolormesh(lon, lat, dif_emis, vmin=-8, vmax=8, transform=ccrs.PlateCarree(), cmap=colormap2)

    shape_feature = ShapelyFeature(Reader(shp_file).geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black')
    axs[0,0].add_feature(shape_feature, linewidth=0.5)
    axs[0,1].add_feature(shape_feature, linewidth=0.5)
    axs[1,0].add_feature(shape_feature, linewidth=0.5)
    axs[1,1].add_feature(shape_feature, linewidth=0.5)


    cbar0=fig.colorbar(c0, ax=axs[0,0], location='bottom', shrink=0.9, aspect=30, pad=0.02,fraction=0.1, format = '%.0f')
    cbar0.set_label('[10$^-$$^8$ mol m$^-$$^2$ s$^-$$^1$]')

    cbar1=fig.colorbar(c1, ax=axs[0,1], location='bottom', shrink=0.9, aspect=30, pad=0.02,fraction=0.1, format = '%.0f')
    cbar1.set_label('[10$^-$$^8$ mol m$^-$$^2$ s$^-$$^1$]')

    cbar2=fig.colorbar(c2, ax=axs[1,0], location='bottom', shrink=0.9, aspect=30, pad=0.02,fraction=0.1, format = '%.0f')
    cbar2.set_label('[10$^-$$^8$ mol m$^-$$^2$ s$^-$$^1$]')

    cbar3=fig.colorbar(c3, ax=axs[1,1], location='bottom', shrink=0.9, aspect=30, pad=0.02,fraction=0.1, format = '%.0f')
    cbar3.set_label('[10$^-$$^8$ mol m$^-$$^2$ s$^-$$^1$]')

    axs[0,0].set_title('Convoluted model-ingested NO$_x$ emissions')
    axs[0,1].set_title('FDA NO$_x$ emissions')
    axs[1,0].set_title('Divergence')
    axs[1,1].set_title('Difference in NO$_x$ emissions [FDA - Model ingested]')

    axs[0,0].text(.01, .99, 'A', ha='left', va='top', size=15, fontweight='bold', transform=axs[0,0].transAxes)
    axs[0,1].text(.01, .99, 'B', ha='left', va='top', size=15, fontweight='bold', transform=axs[0,1].transAxes)
    axs[1,0].text(.01, .99, 'C', ha='left', va='top', size=15, fontweight='bold', transform=axs[1,0].transAxes)
    axs[1,1].text(.01, .99, 'D', ha='left', va='top', size=15, fontweight='bold', transform=axs[1,1].transAxes)


    plt.savefig(save_path, dpi = 300, bbox_inches='tight')
    
    return print('Average plot generated in : ' + save_path)

#%%
def fda_metrics(emis_fda, emis_mod):
    
    from scipy.signal import convolve2d
    psf = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
    conv_emis_mod = convolve2d(emis_mod, psf, 'same')
    
    # Flatten the matrix. Helps to calculate correlation coeficients later
    flat_emis_mod = emis_mod.flatten()
    flat_conv_emis_mod = conv_emis_mod.flatten()
    flat_emis_fda = emis_fda.flatten()

    #Put nan where fda has no emissions
    flat_emis_mod[np.isnan(flat_emis_fda)] = np.nan
    flat_conv_emis_mod[np.isnan(flat_emis_fda)] = np.nan
    
    #Compute metrics for the entire domain
    MB = np.nansum(flat_emis_fda - flat_emis_mod) / np.nansum(flat_emis_mod) * 100
    NMB = np.nansum(np.absolute(flat_emis_fda - flat_emis_mod)) / np.nansum(flat_emis_mod) * 100
    r = np.ma.corrcoef(np.ma.masked_invalid(flat_emis_fda), np.ma.masked_invalid(flat_emis_mod))
    print('**********************************************************************')
    print('Desciption............| NMB | NMGE | R')
    print(['Full domain - Original:', np.round(MB, decimals = 1), np.round(NMB, decimals = 1), np.round(r[0,1], decimals = 2)])

    #Compute metrics for the entire domain
    MB = np.nansum(flat_emis_fda - flat_conv_emis_mod) / np.nansum(flat_conv_emis_mod) * 100
    NMB = np.nansum(np.absolute(flat_emis_fda - flat_conv_emis_mod)) / np.nansum(flat_conv_emis_mod) * 100
    r = np.ma.corrcoef(np.ma.masked_invalid(flat_emis_fda), np.ma.masked_invalid(flat_conv_emis_mod))
    print(['Full domain - Convolut:', np.round(MB, decimals = 1), np.round(NMB, decimals = 1), np.round(r[0,1], decimals = 2)])


    #---------------------------------------------------------------------------------------------------------------------------
    flat_emis_mod[flat_emis_mod<np.nanpercentile(flat_emis_mod, 90)] = np.nan
    flat_conv_emis_mod[np.isnan(flat_emis_mod)] = np.nan
    flat_emis_fda[np.isnan(flat_emis_mod)] = np.nan

    #Compute metrics for the entire domain
    MB = np.nansum(flat_emis_fda - flat_emis_mod) / np.nansum(flat_emis_mod) * 100
    NMB = np.nansum(np.absolute(flat_emis_fda - flat_emis_mod)) / np.nansum(flat_emis_mod) * 100
    r = np.ma.corrcoef(np.ma.masked_invalid(flat_emis_fda), np.ma.masked_invalid(flat_emis_mod))
    print(['Hotspots - Original:', np.round(MB, decimals = 1), np.round(NMB, decimals = 1), np.round(r[0,1], decimals = 2)])


    #Compute metrics for the entire domain
    MB = np.nansum(flat_emis_fda - flat_conv_emis_mod) / np.nansum(flat_conv_emis_mod) * 100
    NMB = np.nansum(np.absolute(flat_emis_fda - flat_conv_emis_mod)) / np.nansum(flat_conv_emis_mod) * 100
    r = np.ma.corrcoef(np.ma.masked_invalid(flat_emis_fda), np.ma.masked_invalid(flat_conv_emis_mod))
    print(['Hotspots - Convolut:', np.round(MB, decimals = 1), np.round(NMB, decimals = 1), np.round(r[0,1], decimals = 2)])
    
    return