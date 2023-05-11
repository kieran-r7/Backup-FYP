import netCDF4
import numpy as np 
import matplotlib.pyplot as plt
import csv
import datetime
import pandas

import sys 
import os 
import glob 
import xarray as xr
import cartopy.crs as ccrs 



path_data_2 = '/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/ERA5/2014_daily_temp'
Dir = os.path.join(path_data_2,'')
files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_01_*.nc')
files_list = [os.path.normpath(i) for i in files_a]


def combine_month(files_list): 
    combined_temp = xr.Dataset()
    global_mean_temp = []
    global_day = []

    #Stage 5 modification 
    global_diurnal_pattern = []
    land_global_mean_temp = []
    land_global_day = [] 

    test = 0 
    for file in files_list: 
        ERA5_day_nc = netCDF4.Dataset(file)
        ERA5_day = xr.open_dataset(file)
        if test == 0: 
            print('/////////////////////////////////////////')
            print('Variables available in the ERA5 file:')
            print(ERA5_day_nc.variables.keys())

            #Dimensions:    (longitude: 144, latitude: 73, time: 8)
            #<xarray.DataArray 't2m' (time: 8, latitude: 73, longitude: 144)>
            print(ERA5_day)

            #Stage 5 modifications below 
            ERA5_daily_pattern = ERA5_day['t2m'].mean(dim=('latitude', 'longitude'))
            # For readability has been converted to degrees celcius. 
            ERA5_daily_pattern -= 273.15
            time_intervals = [0,3,6,9,12,15,18,21]

            fig, ax1 = plt.subplots()
            ax1.plot(time_intervals, ERA5_daily_pattern, linewidth=2, markersize=3, label='Daily Temp')
            ax1.set_xlabel('Time of Day UTC')
            ax1.set_ylabel('Mean average 2m Temperature')
            ax1.set_title('Single Day Average Diurnal Temp')
            plt.show()

            global_diurnal_pattern.append(ERA5_daily_pattern)


            #Initial Stage 4 
            ERA5_daily_mean = ERA5_day.resample(time='D').mean()
            t2m_daily_mean = ERA5_daily_mean['t2m']
            global_mean = t2m_daily_mean.mean(dim=('longitude', 'latitude'))
            global_mean_temp.append(global_mean.values[0]- 273.15)
            global_day.append(t2m_daily_mean['time'].values[0])

            print('Single Day global average temp')
            print(global_mean_temp[0])
            print('date of: ') 
            print(global_day[0])

            combined_temp = t2m_daily_mean
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))

            t2m_daily_mean.plot(ax=ax, cmap='jet',
                            transform=ccrs.PlateCarree())
            ax.coastlines()
            plt.show()

            test = 1
            continue
        ERA5_daily_mean = ERA5_day.resample(time='D').mean()
        t2m_daily_mean = ERA5_daily_mean['t2m']
        global_mean = t2m_daily_mean.mean(dim=('longitude', 'latitude'))
        global_mean_temp.append(global_mean.values[0]- 273.15)
        global_day.append(t2m_daily_mean['time'].values[0])
        combined_temp = xr.concat([combined_temp, t2m_daily_mean], dim="time")

        #Stage 5 modification 
        ERA5_daily_pattern = ERA5_day['t2m'].mean(dim=('latitude', 'longitude'))
        ERA5_daily_pattern -= 273.15
        global_diurnal_pattern.append(ERA5_daily_pattern)

        

    #As the loop is completed sorting and structuring of the variables take place. 
    global_data = list(zip(global_day, global_mean_temp))
    global_data_sorted = sorted(global_data, key=lambda x: x[0])

    combined_temp = combined_temp.sortby('time')

    return(combined_temp,global_data_sorted)



def save_2_file(combined_temp,global_data_sorted,month): 
    combined_temp.to_netcdf(''+month+'_combined.nc')

    with open(''+month+'_Global.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date', 'Global_mean_temp'])
        for row in global_data_sorted:
            writer.writerow(row)

    return 

fig = plt.figure(figsize=[12,5])
combined_temp_jan,global_data_jan = combine_month(files_list)
#save_2_file(combined_temp_jan,global_data_jan,'JAN')
#print(global_data_jan)


'''
fig = plt.figure(figsize=[12,5])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
combined_temp_jan[0].plot(ax=ax, cmap='jet',
                        transform=ccrs.PlateCarree())
ax.coastlines()
plt.show()


#Now all that is needed is repeating for the rest of the months and comment out the two above fuctions. 

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_05_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_may,global_data_may = combine_month(files_list)
save_2_file(combined_temp_may,global_data_may,'MAY')
#05

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_06_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_jun,global_data_jun = combine_month(files_list)
save_2_file(combined_temp_jun,global_data_jun,'JUN')
#06

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_07_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_jul,global_data_jul = combine_month(files_list)
save_2_file(combined_temp_jul,global_data_jul,'JUL')
#07

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_08_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_aug,global_data_aug = combine_month(files_list)
save_2_file(combined_temp_aug,global_data_aug,'AUG')
#08

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_09_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_sep,global_data_sep = combine_month(files_list)
save_2_file(combined_temp_sep,global_data_sep,'SEP')
#09

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_10_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_oct,global_data_oct = combine_month(files_list)
save_2_file(combined_temp_oct,global_data_oct,'OCT')
#10

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_11_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_nov,global_data_nov = combine_month(files_list)
save_2_file(combined_temp_nov,global_data_nov,'NOV')
#11

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_12_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_dec,global_data_dec = combine_month(files_list)
save_2_file(combined_temp_dec,global_data_dec,'DEC')
#12

'''
















print('Script Completed ')