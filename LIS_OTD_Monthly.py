'''
Aim of this script is to move from high to low resolution but focus on a much longer and continuous time period. 
Use of OTD and LIS (TRMM) combined data set is used. 

File chosen was the LRMTS - Low Resolution Monthly Time Series. 

Downloaded from earthdata NASA resource. 
Saved locally in /Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/LRMTS_LIS_OTD


"The product is a 2.5 deg x 2.5 deg gridded composite of Monthly time-series of total (IC+CG) lightning bulk production,
expressed as a flash rate density (fl/km^2/day).  Separate gridded time series from the 5-yr OTD (4/95-3/00) and 
8-yr LIS (1/98-12/05) missions are included, as well as a combined OTD+LIS product.  Lowpass temporal filtering 
(110-day for OTD, 98-day for LIS, 110-day for combined) and spatial moving average filtering (7.5 deg) have been 
applied, as well as best-available detection efficiency corrections and instrument cross-normalizations, as of the 
product generation date (9/01/06)."

Short Description - https://ghrc.nsstc.nasa.gov/uso/ds_docs/lis_climatology/lolrmts_dataset.html

'''
#Import libraries - copied from previous script
import netCDF4
import numpy as np 
import matplotlib.pyplot as plt
import csv
import datetime
import pandas
import xarray as xr 
import cartopy.crs as ccrs

file_name = 'LISOTD_LRMTS_V2.3.2015.nc'
path_data = '/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/LRMTS_LIS_OTD/'
file_path =  path_data + file_name

LRMTS_dataframe = xr.open_dataset(file_path,decode_times=False)

print('TEST No1: ')
print(LRMTS_dataframe)
print(LRMTS_dataframe['LRMTS_COM_FR'])

fig = plt.figure(figsize=[12,5])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))

#Can show month at a time - issue is that the early months are empty. 
LRMTS_dataframe['LRMTS_COM_FR'][:,:,120].plot(ax=ax, cmap='jet',
                   transform=ccrs.PlateCarree())
ax.coastlines()

plt.show()




# Extract the LRMTS_COM_FR variable
lrmts = LRMTS_dataframe['LRMTS_COM_FR']

# Calculate the global flash density for each month
global_flash_density_per_month = []
months_since_jan_95 = []
for i in range(240):
    lrmts_month = lrmts.isel(Month_since_Jan_95=i)
    num_non_nan = np.sum(~np.isnan(lrmts_month))
    if num_non_nan > 0:
        global_flash_density = lrmts_month.sum(dim=('Latitude', 'Longitude')) / num_non_nan
    else:
        global_flash_density = 0.0
    global_flash_density_per_month.append(global_flash_density)
    months_since_jan_95.append(i)

# Create a scatter plot
fig, ax = plt.subplots()
#ax.plot(months_since_jan_95, global_flash_density_per_month)


ax.plot(months_since_jan_95[:5], global_flash_density_per_month[:5], color='blue')
ax.plot(months_since_jan_95[6:62], global_flash_density_per_month[6:62],color='red')
ax.plot(months_since_jan_95[62:228], global_flash_density_per_month[62:228])
ax.plot(months_since_jan_95[230:], global_flash_density_per_month[230:], color='blue')

for i in range(0, len(months_since_jan_95), 12):
    ax.axvline(x=months_since_jan_95[i], linestyle='--', linewidth=0.5, color='gray')

# Set the x-axis label
ax.set_xticks(np.arange(0, len(months_since_jan_95), 12))
ax.set_xlabel('Months since Jan 1995')

# Set the y-axis label
ax.set_ylabel('Global flash density (fl/km^2/day)')
plt.title('OTD/LIS LRMTS global flash density from 1995 to 2015')
plt.show()
# Show the plot


GISTEMP_df = pandas.read_csv('/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/GISTEMP/graph.csv')
fig2, ax2 = plt.subplots()
ax2.plot(GISTEMP_df['Months_after_Jan95'], GISTEMP_df['Land+Ocean'], color='green', label='Temperature (Land+Ocean)')
ax2.plot(GISTEMP_df['Months_after_Jan95'], GISTEMP_df['Land_Only'], color='orange', label='Temperature (Land Only)',alpha = 0.5,linestyle='--')
ax2.plot(GISTEMP_df['Months_after_Jan95'], GISTEMP_df['Open_Ocean'], color='yellow', label='Temperature (Open Ocean)',alpha = 0.5,linestyle='-.')


#calculate equation for trendline
z = np.polyfit(GISTEMP_df['Months_after_Jan95'],GISTEMP_df['Land+Ocean'] , 1)
p = np.poly1d(z)

plt.title('GISTEMP monthly mean Surface Temperature data' )

ax2.plot(GISTEMP_df['Months_after_Jan95'], p(GISTEMP_df['Months_after_Jan95']), label = 'Trend line for Temperature change')

# Add legend and labels
ax2.legend(loc='upper right')
ax2.set_xticks(np.arange(0, len(months_since_jan_95), 12))
ax2.set_xlabel('Months since Jan 1995')
ax2.set_ylabel('Temperature Anomaly w.r.t. 1951-80(°C)')

# Show the plot
plt.show()








