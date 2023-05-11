from curses import init_pair
import netCDF4
import numpy as np 
import matplotlib.pyplot as plt
import csv
import datetime

import pandas


##Input Data - file paths saved locally 
file_name = 'ISS_LIS_SC_V2.1_20210107_005404_FIN.nc'
path_data = '/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/LIS_ISS_QC/2021-0107/'
file_path =  path_data + file_name


# These are the desired variables 
hourly_flash_count = pandas.DataFrame()
acc_flash_count = pandas.DataFrame()


#Set up the dataframes with desired headers : This is easier to use once in loops. 
#hourly_flash_count(columns=['dates','hour'])
#acc_flash_count(columns=['dates','acc_count'])

def tai93_to_datetime(tai93_time):
    base_tai93 = pandas.Timestamp("1993-01-01T00:00:00")
    converted_time = base_tai93 + pandas.Timedelta(seconds=tai93_time)
    return converted_time


counter = 0
hours_list = []

for i in range(1, 8):
    # read the file into a pandas dataframe
    file_name_loop = "PART2_Orbit1_0"+str(i)+"_01_2021.csv"
    path_flash = "/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/LIS_Week/"
    t = path_flash+file_name_loop

    #specify rows to import
    specific_rows = [0,1]
    #import specific rows from CSV into DataFrame
    orbit_values_r2 = pandas.read_csv(t, skiprows = lambda x: x not in specific_rows)
    flash_data = pandas.read_csv(t,skiprows=3)
    origin = orbit_values_r2['Orbit Start']
    flash_data['TA19 Start Time'] = flash_data['Start Time'] + origin[0]
    flash_data['TA19 Start Time'] = flash_data['TA19 Start Time'].apply(tai93_to_datetime)

    dates = flash_data['TA19 Start Time']
    temp_hour = flash_data['TA19 Start Time'].dt.hour

    hours_list.extend(temp_hour + counter)
    print(i)
    counter += 24

# create dataframe with hours_list as a new column

hours = np.arange(24*7)
hourly_flash_count = pandas.DataFrame({'hours': pandas.Series(hours_list)})
hourly_bar = pandas.Series(hours_list).value_counts().sort_index()

print(hourly_flash_count)
print('.csv file read in')

plt.bar(hourly_bar.index, hourly_bar)
plt.xticks(np.arange(0, 24*7, 12))
plt.xlabel('Hour (UTC)')
plt.ylabel('Count')
plt.show()