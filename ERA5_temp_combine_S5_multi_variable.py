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
files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_11_*.nc')
files_list = [os.path.normpath(i) for i in files_a]


def combine_month(files_list): 
    combined_temp = xr.Dataset()
    combined_humidity = xr.Dataset()
    global_mean_temp = []
    global_day = []

    #Stage 5 modification 
    global_diurnal_pattern = []
    global_humidity_pattern = []
    global_RH_pattern = []

    g_d2m_temp = []
    g_RH_temp = []
    dew_point_day = [] 

    land_global_mean_temp = []
    land_global_day = [] 

    test = 0 
    for file in files_list: 
        date_string = file.split('/')[-1].split('_')[-3:]
        date_string = '_'.join(date_string)

        ERA5_day_nc = netCDF4.Dataset(file)
        ERA5_day = xr.open_dataset(file)

        ERA5_humidity_nc = netCDF4.Dataset('/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/ERA5/2014_daily_dewpoint/era5_daily_2m_dewpoint_temperature_' + date_string)
        ERA5_humidity = xr.open_dataset('/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/ERA5/2014_daily_dewpoint/era5_daily_2m_dewpoint_temperature_' + date_string)

        if test == 0: 
            print('/////////////////////////////////////////')
            print('Variables available in the ERA5 file:')
            print(ERA5_day_nc.variables.keys())
            print(ERA5_humidity_nc.variables.keys())

            #Dimensions:    (longitude: 144, latitude: 73, time: 8)
            #<xarray.DataArray 't2m' (time: 8, latitude: 73, longitude: 144)>
            print(ERA5_day['t2m'])
            print("")
            print(ERA5_humidity['d2m'])
            '''

            fig = plt.figure(figsize=[12,5])
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))

            ERA5_humidity['d2m'][0,:,:].plot(ax=ax, cmap='jet',
                            transform=ccrs.PlateCarree())
            ax.coastlines()
            plt.title('08-01-2014 Hr 0 [2m_dewpoint_temperature]')
            plt.show()
            '''

            #Formula modification to find the relative humidity 
            #Constants in the August-Roche-Magnus approximation
            a  = 17.625
            b = 243.04

            T = ERA5_day['t2m'].values - 273.15  # converting from Kelvin to Celsius
            Td = ERA5_humidity['d2m'].values - 273.15

            RH = 100 * (np.exp((a * Td)/(b + Td)) / np.exp((a * T)/(b + T)))

            RH_daily_pattern = np.mean(RH, axis=(1, 2))
            RH_daily_mean = np.mean(RH_daily_pattern)

            global_RH_pattern.append(RH_daily_pattern)
            g_RH_temp.append(RH_daily_mean)
            

            #Stage 5 modifications below 
            ERA5_daily_pattern = ERA5_day['t2m'].mean(dim=('latitude', 'longitude'))
            humidity_pattern = ERA5_humidity['d2m'].mean(dim=('latitude', 'longitude'))

            # For readability has been converted to degrees celcius. 
            ERA5_daily_pattern -= 273.15
            humidity_pattern -= 273.15
            time_intervals = [0,3,6,9,12,15,18,21]

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(time_intervals, ERA5_daily_pattern, linewidth=2, markersize=3, label='Daily Temp')
            ax2.plot(time_intervals, RH_daily_pattern, linewidth=2, markersize=3, label='RH_daily_pattern',color = 'red')

            ax1.set_xlabel('Time of Day UTC')
            ax1.set_ylabel('Mean average 2m Temperature')

            ax2.set_ylabel('Mean average Relative humidity')
            ax1.set_title('Single Day Average Diurnal Temp')
            plt.show()

            #Ready for the loop 
            global_diurnal_pattern.append(ERA5_daily_pattern.values)
            global_humidity_pattern.append(humidity_pattern.values)


            #Stage 4 - Annual patterns 
            ERA5_daily_mean = ERA5_day.resample(time='D').mean()
            t2m_daily_mean = ERA5_daily_mean['t2m']
            global_mean = t2m_daily_mean.mean(dim=('longitude', 'latitude'))
            global_mean_temp.append(global_mean.values[0]- 273.15)
            global_day.append(t2m_daily_mean['time'].values[0])

            #Repeated for humidity and dew_point temperature 
            dewpoint_daily_mean = ERA5_humidity.resample(time='D').mean()
            d2m_daily_mean = dewpoint_daily_mean['d2m']
            g_m_d2m = d2m_daily_mean.mean(dim=('longitude', 'latitude'))
            g_d2m_temp.append(g_m_d2m.values[0]- 273.15)
            dew_point_day.append(d2m_daily_mean['time'].values[0])

            

            print('Single Day global average temp')
            print(global_mean_temp[0])
            print('date of: ') 
            print(global_day[0])

            print('Check that the dew_point calculation has been done properly')
            print('Single Day global average temp')
            print(g_d2m_temp[0])
            print('date of: ') 
            print(dew_point_day[0])


            combined_temp = t2m_daily_mean
            fig, ax = plt.subplots()
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
        global_diurnal_pattern.append(ERA5_daily_pattern.values)


        dewpoint_daily_mean = ERA5_humidity.resample(time='D').mean()
        d2m_daily_mean = dewpoint_daily_mean['d2m']
        g_m_d2m = d2m_daily_mean.mean(dim=('longitude', 'latitude'))
        g_d2m_temp.append(g_m_d2m.values[0]- 273.15)

        humidity_pattern = ERA5_humidity['d2m'].mean(dim=('latitude', 'longitude'))
        humidity_pattern -= 273.15
        global_humidity_pattern.append(humidity_pattern.values)
  

        T = ERA5_day['t2m'].values - 273.15  # converting from Kelvin to Celsius
        Td = ERA5_humidity['d2m'].values - 273.15
        RH = 100 * (np.exp((a * Td)/(b + Td)) / np.exp((a * T)/(b + T)))

        RH_daily_pattern = np.mean(RH, axis=(1, 2))
        RH_daily_mean = np.mean(RH_daily_pattern)
        global_RH_pattern.append(RH_daily_pattern)
        g_RH_temp.append(RH_daily_mean)

    #As the loop is completed sorting and structuring of the variables take place. 
    global_data = list(zip(global_day, global_mean_temp, global_diurnal_pattern))
    global_data_sorted = sorted(global_data, key=lambda x: x[0])

    pattern_data = list(zip(global_day, global_diurnal_pattern))
    pattern_sorted = sorted(pattern_data, key=lambda x: x[0])

    humidity_data = list(zip(global_day,g_d2m_temp, global_humidity_pattern,g_RH_temp,global_RH_pattern))
    humidity_sorted = sorted(humidity_data, key=lambda x: x[0])


    combined_temp = combined_temp.sortby('time')

    return(combined_temp,global_data_sorted,pattern_sorted,humidity_sorted)


def combine_month_new_v(files_list): 

    new_var_day = []
    new_var_pattern = []
    extract_day = [] 

    test = 0 
    for file in files_list: 
        print(file)
        date_string = file.split('/')[-1].split('_')[-3:]
        date_string = '_'.join(date_string)

        ERA5_new_V_day_nc = netCDF4.Dataset(file)
        ERA5_new_V_day = xr.open_dataset(file)

        if test == 0: 
            print('/////////////////////////////////////////')
            print('Variables available in the ERA5 file:')
            print(ERA5_new_V_day_nc.variables.keys())

            #Dimensions:    (longitude: 144, latitude: 73, time: 8)
            #<xarray.DataArray 't2m' (time: 8, latitude: 73, longitude: 144)>
            print(ERA5_new_V_day['mvimd'])
            
            fig = plt.figure(figsize=[12,5])
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))

            ERA5_new_V_day['mvimd'][0,:,:].plot(ax=ax, cmap='jet',
                            transform=ccrs.PlateCarree())
            ax.coastlines()
            plt.title('File test')
            plt.show()
            
            var_daily_mean = ERA5_new_V_day.resample(time='D').mean()
            cape_daily_mean = var_daily_mean['mvimd']

            cape_1 = cape_daily_mean.mean(dim=('longitude', 'latitude'))
            new_var_day.append(cape_1.values[0])
            extract_day.append(cape_daily_mean['time'].values[0])

            a_12 = ERA5_new_V_day['mvimd'].mean(dim=('latitude', 'longitude'))
            new_var_pattern.append(a_12.values)
            time_intervals = [0,3,6,9,12,15,18,21]

            fig, ax1 = plt.subplots()
            ax1.plot(time_intervals, new_var_pattern[0], linewidth=2, markersize=3, label='Daily Temp')

            ax1.set_xlabel('Time of Day UTC')
            ax1.set_ylabel('new_variable')

            ax1.set_title('Single Day Average Diurnal Temp')
            plt.show()

            print('Check that the dew_point calculation has been done properly')
            print('Single Day global average temp')
            print(new_var_day[0])
            print('date of: ') 
            print(extract_day[0])



            test = 1
            continue

        var_daily_mean = ERA5_new_V_day.resample(time='D').mean()
        cape_daily_mean = var_daily_mean['mvimd']

        cape_1 = cape_daily_mean.mean(dim=('longitude', 'latitude'))
        new_var_day.append(cape_1.values[0])
        extract_day.append(cape_daily_mean['time'].values[0])
        a_12 = ERA5_new_V_day['mvimd'].mean(dim=('latitude', 'longitude'))
        new_var_pattern.append(a_12.values)


    #As the loop is completed sorting and structuring of the variables take place. 
    pattern_data = list(zip(extract_day,new_var_day, new_var_pattern))
    pattern_sorted = sorted(pattern_data, key=lambda x: x[0])

    return(pattern_sorted)
    


def save_2_file(combined_temp,global_data_sorted,month): 
    combined_temp.to_netcdf(''+month+'_combined.nc')

    with open(''+month+'_Global.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date', 'Global_mean_temp','Diurnal_pattern'])
        for row in global_data_sorted:
            writer.writerow(row)

    return 


def save_2_file_s5(combined_temp,global_data_sorted,month): 
    combined_temp.to_netcdf(''+month+'_combined.nc')

    with open(''+month+'_Global.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date', 'Diurnal_pattern'])
        for row in global_data_sorted:
            writer.writerow(row)

    return 



def save_2_file_humidity(humidity_sorted,month): 
    
    with open(''+month+'_Global_Humidity.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(['Date','d2m_mean', 'd2m_pattern','RH_mean', 'RH_pattern'])
        #global_day,g_d2m_temp, global_humidity_pattern,g_RH_temp,global_RH_pattern))
        for row in humidity_sorted:
            writer.writerow(row)
    return 

def save_2_new_var(pattern_sorted,month,unit): 
    
    with open(''+month+'_'+unit+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(['Date', unit+'_mean', unit+'_pattern'])
        #global_day,g_d2m_temp, global_humidity_pattern,g_RH_temp,global_RH_pattern))
        for row in pattern_sorted:
            writer.writerow(row)
    return 

'''
#combined_temp_jan,global_data_jan,pattern_sorted,humidity_sorted = combine_month(files_list)
path_data_2 = '/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/ERA5/2014_daily_CAPE'
Dir = os.path.join(path_data_2,'')
files_a = glob.glob(Dir+'era5_daily_Convective_available_potential_energy_2014_12_*.nc')
#era5_daily_Convective_available_potential_energy_2014_01_01.nc
files_list = [os.path.normpath(i) for i in files_a]

pattern_sorted=combine_month_new_v(files_list)
save_2_new_var(pattern_sorted,'DEC','cape')
'''

path_data_2 = '/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/ERA5/2014_daily_MVIMD'
Dir = os.path.join(path_data_2,'')
files_a = glob.glob(Dir+'era5_daily_mean_vertically_integrated_moisture_divergence_2014_12_*.nc')
#era5_daily_Convective_available_potential_energy_2014_01_01.nc
files_list = [os.path.normpath(i) for i in files_a]

pattern_sorted=combine_month_new_v(files_list)
save_2_new_var(pattern_sorted,'DEC','mvimd')

#print(humidity_sorted)
#save_2_file_humidity(humidity_sorted,'OCT')
#print(global_data_jan)


'''
#Now all that is needed is repeating for the rest of the months and comment out the two above fuctions. 
files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_02_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_may,ignore,global_data_feb = combine_month(files_list)
save_2_file(combined_temp_may,global_data_feb,'FEB_S5')
#02

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_03_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_may,ignore,global_data_mar = combine_month(files_list)
save_2_file(combined_temp_may,global_data_mar,'MAR_S5')
#03

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_04_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_may,ignore,global_data_apr = combine_month(files_list)
save_2_file(combined_temp_may,global_data_apr,'APR_S5')
#04

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_05_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_may,ignore,global_data_may = combine_month(files_list)
save_2_file(combined_temp_may,global_data_may,'MAY_S5')
#05

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_06_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_jun,ignore,global_data_jun = combine_month(files_list)
save_2_file(combined_temp_jun,global_data_jun,'JUN_S5')
#06

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_07_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_jul,ignore,global_data_jul = combine_month(files_list)
save_2_file(combined_temp_jul,global_data_jul,'JUL_S5')
#07

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_08_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_aug,ignore,global_data_aug = combine_month(files_list)
save_2_file(combined_temp_aug,global_data_aug,'AUG_S5')
#08

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_09_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_sep,ignore,global_data_sep = combine_month(files_list)
save_2_file(combined_temp_sep,global_data_sep,'SEP_S5')
#09

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_10_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_oct,ignore,global_data_oct = combine_month(files_list)
save_2_file(combined_temp_oct,global_data_oct,'OCT_S5')
#10

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_11_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_nov,ignore,global_data_nov = combine_month(files_list)
save_2_file(combined_temp_nov,global_data_nov,'NOV_S5')
#11

files_a = glob.glob(Dir+'era5_daily_2m_temperature_2014_12_*.nc')
files_list = [os.path.normpath(i) for i in files_a]

fig = plt.figure(figsize=[12,5])
combined_temp_dec,ignore,global_data_dec = combine_month(files_list)
save_2_file(combined_temp_dec,global_data_dec,'DEC_S5')
#12


'''















print('Script Completed ')