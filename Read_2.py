#This script will attempt to read in data sets from NASA 
#https://github.com/adityapatel205/geodac-2022/blob/main/lis_tropomi_grouped_analysis/nb01_iss_lis_download-files_userinput_time_region_ncfile_filtered_v3.ipynb
#https://pypi.org/project/earthdata/

from tracemalloc import start
import earthaccess
from earthaccess import Auth, Store, DataCollections, DataGranules
import netCDF4
import numpy as np 
### AUTHENTICATION STEP

auth = Auth()
auth.login(strategy="/Users/admin/Documents/FInal_Semester_Project/Python/Login/login.nc")
# are we authenticated?
print(auth.authenticated)


##Test 1 Single netCDF file 
##Using the QC ISS LIS Dataset 01/01/21 has been selected and the aim is to produce a file with all the flashes detected in
##single day. 

#Online directory with all the data needed: 
##https://ghrc.nsstc.nasa.gov/pub/lis/iss/data/science/final/nc/2021/0101/

import matplotlib.pyplot as plt
import csv
import datetime
#For testing only needed if looking at scripts in development and for understanding of readibility
import time

#from itertools import izip
#import glob
#Will need to import these but will come back and look at this when needed - Terminal required. 

##Input Data - file paths saved locally 
file_name = 'ISS_LIS_SC_V2.1_20210401_005059_FIN.nc'
path_data = '/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/LIS_ISS_QC/2021-0401/'
file_path =  path_data + file_name

##Test 1 
#Access and read a netCDF formatted file. 
f = netCDF4.Dataset(file_path)
#Create variable for the start and stop times of the orbit in the file. 
orbit_id = []
orbit_start = []
orbit_end = []
orbit_duration = []
start_value = f.variables['orbit_summary_TAI93_start'][:].data.tolist()
#start_value_units = f.variables['orbit_summary_TAI93_start']
end_value = f.variables['orbit_summary_TAI93_end'][:].data.tolist()
#end_value_units = f.variables['orbit_summary_TAI93_end']

#Identifier (easier than full file name)
orbit_id = f.variables['orbit_summary_id_number'][:].data.tolist()
orbit_start.append(start_value)
orbit_end.append(end_value)
orbit_duration.append(end_value - start_value)
# Data extracted: start_value = 883616146.3  // end_value = 883621719.3
print('Test 1.0 Complete ')
convert = str(datetime.timedelta(seconds = orbit_duration[0]))
print('Orbit ID')
print(orbit_id)
print('Orbit Duration')
print(convert)

##Test 1.1 
#Lightning location to be found and wrote to a csv file. - i.e make readable
#Create empty arrays to for latitude and longitude values of each flash

lat_lgn = np.array([]) #latitude
lon_lgn = np.array([]) #longitude
start_lgn = np.array([]) #start time
rad_lgn = np.array([]) #radiance
d_index_lgn = np.array([]) #density index
flag_lgn = np.array([]) #alert flag 
c_index_lgn = np.array([]) #cluster index
#Add to the array - this is a better format if this is used to store info on each day rather than each orbit. 
lat_lgn = np.concatenate([lat_lgn,f.variables['lightning_flash_lat'][:]])
lon_lgn = np.concatenate([lon_lgn,f.variables['lightning_flash_lon'][:]]) 
print('Test 1.1 Complete ')
print('No. of flashes/lightning strikes detected: ')
#This can be any of the variables but the length can be used as a verification of sucessful loading of the file. 
print(len(lat_lgn))
#Stored data for each variable that we want to investigate: 

#Essential lightning data from file.
start_lgn = f.variables['lightning_flash_TAI93_time'][:].data.tolist()
#Lightning Behaviour analysis 
rad_lgn = f.variables['lightning_flash_radiance'][:].data.tolist()
d_index_lgn = f.variables['lightning_flash_density_index'][:].data.tolist()
#Quality Assessment/Control 
flag_lgn = f.variables['lightning_flash_alert_flag'][:].data.tolist()
c_index_lgn = f.variables['lightning_flash_cluster_index'][:].data.tolist()
'''
Reference: NASA LIS ISS Data User Guide: https://ghrc.nsstc.nasa.gov/pub/lis/iss/doc/isslis_dataset.pdf

'''
#Alert Flag ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#Changes each value in the alert flag back to a binary value, this is then used to check for any of the fatal flags being a 1

def alert_to_bin(flag_list):
    binary_list = []
    for flag in flag_list:
        binary = bin(flag)[2:].zfill(8) # Makes sure it is 8bit 
        binary_list.append([int(bit) for bit in binary]) # convert binary string to list of integers
    return binary_list

def alert_flags_1(flag_list):
    binary_list = alert_to_bin(flag_list)
    binary_array = np.array(binary_list)
    bit_mask = np.array([1,0,1,0,1,0,1,0]) # Define the bit mask
    masked_array = np.bitwise_and(binary_array, bit_mask.reshape(1, -1)) # Apply the bit mask to the binary values
    fatal_flag = np.any(masked_array[:, [1, 3, 5, 7]] == 1, axis=1) # Check if any of the specified bits are set to 1
    return fatal_flag.astype(int).tolist()

#This no has a single binary value if a fatal flag was 'on'
#Now the quality filter any flash that has a 1 value here will not be taken into the anlysis. 
fatal_flag = alert_flags_1(flag_lgn)

#Cluster Index ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Create cluster_warning flag using list comprehension
cluster_warning = [0 if c_index > 10 else 1 for c_index in c_index_lgn]

#Radiance ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
factor = 1e-6
# Convert units to J/sr/m^2/um - easier readability when graphed. 
rad_lgn = np.array(rad_lgn) * factor

#Density Index  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
#Need to check multiple files before any further analysis is done. 

#start of lightning flash   //////////////////////////////////////////////////////////////////////////////////////////////////////////////
#For better understanding and easier plotting of graphs and trends. 
#The origin for flash data will be the orbit start time. 
start_lgn_origin = [x - orbit_start[0] for x in start_lgn]

output_file = 'PART2_Orbit1_01_04_2021.csv'
Orbit_header = ['Orbit ID', 'Orbit Start', 'Orbit End', 'Orbit Duration']
Flash_header = ['Latitude', 'Longitude', 'Start Time', 'Radiance', 'Density Index', 'Alert_Flag', 'Cluster Index','Cluster Warning Flag']

# Open output file and write headers
with open(output_file, 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(Orbit_header)
    writer.writerow([orbit_id,start_value,end_value,orbit_duration[0]])
    writer.writerow([])

    writer.writerow(Flash_header)
    for i in range (len(lat_lgn)): 
        writer.writerow([lat_lgn[i],lon_lgn[i],start_lgn_origin[i],rad_lgn[i],d_index_lgn[i],fatal_flag[i],c_index_lgn[i],cluster_warning[i]])

f_out.close()
print('Script part 1 completed')
# Start of Part 2 

import sys 
import os 
import glob 


path_data_2 = '/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/LIS_ISS_QC/2021-0401/Orbits'
Dir = os.path.join(path_data_2,'')
files_list = glob.glob(Dir+'ISS_LIS_*.nc')
files_part2 = [os.path.normpath(i) for i in files_list]
print('Loaded')

def variable_extract(data_set,orbit_start,orbit_end,output_file): 
    #May remove: 
    end_value = data_set.variables['orbit_summary_TAI93_end'][:].data.tolist()
    orbit_end.append(end_value)
    start_value = data_set.variables['orbit_summary_TAI93_start'][:].data.tolist()
    orbit_start.append(start_value)
    #-- CLEAR THE BELOW
    lat_lgn = np.array([]) #latitude
    lon_lgn = np.array([]) #longitude
    start_lgn = np.array([]) #start time
    rad_lgn = np.array([]) #radiance
    d_index_lgn = np.array([]) #density index
    flag_lgn = np.array([]) #alert flag 
    c_index_lgn = np.array([]) #cluster index
    #-- READ IN THE FOLLOWING 
    if 'lightning_flash_lat' not in data_set.variables:
        print("Error: Variable 'lightning_flash_lat' not found in NetCDF file.")
        return orbit_start, orbit_end

    lat_lgn = np.concatenate([lat_lgn,data_set.variables['lightning_flash_lat'][:]])
    lon_lgn = np.concatenate([lon_lgn,data_set.variables['lightning_flash_lon'][:]]) 
    start_lgn = data_set.variables['lightning_flash_TAI93_time'][:].data.tolist()
    rad_lgn = data_set.variables['lightning_flash_radiance'][:].data.tolist()
    d_index_lgn = data_set.variables['lightning_flash_density_index'][:].data.tolist()
    flag_lgn = data_set.variables['lightning_flash_alert_flag'][:].data.tolist()
    c_index_lgn = data_set.variables['lightning_flash_cluster_index'][:].data.tolist()
    fatal_flag = alert_flags_1(flag_lgn)

    cluster_warning = [0 if c_index > 10 else 1 for c_index in c_index_lgn]
    factor = 1e-6
    rad_lgn = np.array(rad_lgn) * factor
    start_lgn_origin = [x - orbit_start[0] for x in start_lgn]

    with open(output_file, 'a', newline='') as f_out:
        writer = csv.writer(f_out)
        for j in range (len(lat_lgn)): 
            writer.writerow([lat_lgn[j],lon_lgn[j],start_lgn_origin[j],rad_lgn[j],d_index_lgn[j],fatal_flag[j],c_index_lgn[j],cluster_warning[j]])
    f_out.close()
    print('Orbit Data added successfully : ')
    return orbit_start, orbit_end



from netCDF4 import Dataset
for i in files_part2: 
    data_set = Dataset(i,'r')
    orbit_start, orbit_end = variable_extract(data_set,orbit_start,orbit_end,output_file)

    a = 2 

print('Script part 2 completed')

#The csv file is created and no can be used as a single day file rather than the need to access multiple orbit netCDF. 

import pandas as pd 

# Load the CSV data into a Pandas DataFrame
day_data = pd.read_csv(output_file, skiprows=3)

# Convert the start time column to a datetime object and sort (This is to make the code more robust incase files are not read in a 
# particular order )
#day_data['Start Time'] = pd.to_datetime(day_data['Start Time'], unit='s')
day_data = day_data.sort_values('Start Time')

# Calculate the accumulated flash count
day_data = day_data.reset_index(drop=True)
day_data['Accumulated Flash Count'] = day_data.index + 1

'''
#Plotting //////////
fig1, (ax0,ax1,ax2) = plt.subplots(3, 1, figsize=(9,9))

ax0.scatter(day_data['Start Time'], day_data['Accumulated Flash Count'],s=5)
ax0.set_xlabel('Time')
ax0.set_ylabel('Accumulated Flash Count')
ax0.set_title('Flash Count - 01-01-2021')

ax1.scatter(day_data['Start Time'], day_data['Radiance'],s=5)
ax1.set_xlabel('Time')
ax1.set_ylabel('Radiance')
ax1.set_title('Radiance - 01-01-2021')

ax2.scatter(day_data['Start Time'], day_data['Cluster Index'],s=5)
ax2.set_xlabel('Time')
ax2.set_ylabel('Cluster Index')
ax2.set_title('Cluster Index - 01-01-2021')

plt.tight_layout()
plt.show()

fig2, (ax3,ax4) = plt.subplots(2, 1, figsize=(9,9))
ax3.hist(day_data['Cluster Index'], bins=50)
ax3.set_xlabel('Cluster Index')
ax3.set_ylabel('Count')
ax3.set_title('Histogram of Cluster Index')

# Calculate mean and standard deviation for Cluster Index
mean_cluster_index = day_data['Cluster Index'].mean()
std_cluster_index = day_data['Cluster Index'].std()

# Add vertical lines for mean and mean ± std
ax3.axvline(x=mean_cluster_index, color='r', linestyle='--', label='mean')
ax3.axvline(x=mean_cluster_index-std_cluster_index, color='b', linestyle='--', label='-1 std.')
ax3.axvline(x=mean_cluster_index+std_cluster_index, color='g', linestyle='--', label='+1 std.')
ax3.legend()

ax4.hist(day_data['Density Index'], bins=50)
ax4.set_xlabel('Density Index')
ax4.set_ylabel('Count')
ax4.set_title('Histogram of Density Index')

mean_di = day_data['Density Index'].mean()
std_di = day_data['Density Index'].std()

# Add vertical lines for mean and mean ± std
ax4.axvline(x=mean_di, color='r', linestyle='--', label='mean')
ax4.axvline(x=mean_di-std_di, color='b', linestyle='--', label='-1 std.')
ax4.axvline(x=mean_di+std_di, color='g', linestyle='--', label='+1 std.')
ax4.legend()

plt.tight_layout()
plt.show()

'''
#Daily analysis 

# load data
bin_edges = np.arange(0, 86400, 3600)
# Group data by hourly period and calculate flash counts
hourly_data = pd.cut(day_data['Start Time'], bins=bin_edges, labels=range(1, 24))
# create a new DataFrame with hourly flash counts
flash_count = pd.DataFrame({'Flash Count': hourly_data.groupby(hourly_data).size()})
# plot the flash counts
flash_count.plot(kind='bar', xlabel='Hourly Period', ylabel='Flash Count')
plt.show()

