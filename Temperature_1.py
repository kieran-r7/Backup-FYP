import os
import cdsapi
import netCDF4
import pandas
import numpy as np
import xarray as xr 
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
import csv
import datetime
#For testing only needed if looking at scripts in development and for understanding of readibility
import time



#Initial test No1. Using a single file that has been downloaded using the CDS API 
#Will be using the 2m temperature data for 01-01-2021 
#ERA5 information assumed to have a 0.25 degree resolution with hourly temperatures. 

##Input Data - file paths saved locally - will need to change from a single variable to a list. 
file_name = 'era5_2m_temperature_2021_01_01.nc'
path_data = '/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/ERA5/Jan2021_2mTemp/'
file_path =  path_data + file_name
f = netCDF4.Dataset(file_path)

dset = xr.open_dataset(file_path)
print(dset)
#This is displays the metadata. Not needed but terminal readablity is a good check for expected contents 
print('////////////////////////////////////////////////////////////////////////////////////////////////////////')
print(dset['t2m'])

fig = plt.figure(figsize=[12,5])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))

dset['t2m'][0,:,:].plot(ax=ax, cmap='jet',
                   transform=ccrs.PlateCarree())
ax.coastlines()
plt.title('01-01-2021 Hr 0 [2m_temperature]')
plt.show()




