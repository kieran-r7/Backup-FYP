#from curses import init_pair
#from pickle import TRUE
import netCDF4
import numpy as np 
import matplotlib.pyplot as plt
import csv
import datetime
import pandas

# These are the desired variables 
hourly_flash_count = pandas.DataFrame()
acc_flash_count = pandas.DataFrame()

def tai93_to_datetime(tai93_time):
    base_tai93 = pandas.Timestamp("1993-01-01T00:00:00")
    converted_time = base_tai93 + pandas.Timedelta(seconds=tai93_time)
    return converted_time

counter = 0
hours_list = []
#A modification is needed as the flash data needs to be combined and kept in a dataframe rather than extracting the hours 
#The hours variable is still used to display the barchart so has not been removed. 
combined_flash = pandas.DataFrame()
for i in range(1, 8):
    # read in individual flash files of each orbit and combine into one dataframe 
    file_name_loop = "PART2_Orbit1_0"+str(i)+"_10_2021.csv"
    path_flash = "/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/LIS_Week_Oct/"
    t = path_flash+file_name_loop

    #These are the rows that contain the orbit headers and the values needed. 
    specific_rows = [0,1]
    orbit_values_r2 = pandas.read_csv(t, skiprows = lambda x: x not in specific_rows)
    #Then 3 rows are skipped so that the flash table is used and stored in the dataframe for a single file 
    flash_data = pandas.read_csv(t,skiprows=3)
    origin = orbit_values_r2['Orbit Start']

    #To have all time variables collected in one the origin of the orbit must be added to the time of occurence
    #This is then converted to a date time variable as this makes it easier to extract the hour parameter. 
    flash_data['TA19 Start Time'] = flash_data['Start Time'] + origin[0]
    flash_data['TA19 Start Time'] = flash_data['TA19 Start Time'].apply(tai93_to_datetime)
    #dates was used to test if the conversion worked correctly
    dates = flash_data['TA19 Start Time']
    temp_hour = flash_data['TA19 Start Time'].dt.hour

    #This is the combined dataframe where each file content is added to the end of each loop iteration. 
    #hours_list contains the hour of each flash occurence with the counter taking into consideration each new day
    combined_flash = combined_flash.append(flash_data, ignore_index=True)
    hours_list.extend(temp_hour + counter)
    print(i)
    counter += 24

# create dataframe with hours_list as a new column

hours = np.arange(24*7)
hourly_flash_count = pandas.DataFrame({'hours': pandas.Series(hours_list)})
hourly_bar = pandas.Series(hours_list).value_counts().sort_index()
#This now has the count of flashes detecected for each hour and a csv produced for diurnal analysis
hourly_bar.to_csv("EXEL_KR_Oct.csv",index = True)
#hourly_flash_count has the count for hours 0-23 
print(hourly_flash_count)
print('.csv file read in')

#bar chart showing the hourly flash count for the week time period 
plt.bar(hourly_bar.index, hourly_bar)
plt.xticks(np.arange(0, 24*7, 12))
plt.xlabel('Hour (UTC)')
plt.ylabel('Count')
plt.title('Hourly Flash group - Jan')
plt.show()


#Start of part 2 - Read in the ERA5 file that has been created 
#Test 1 for the first hour of 01-01-2021
#Section copied from the graph creation in Temperature_1.py 
import xarray as xr 

''''''

def com_temp_files(combined_2m_temp):
    #similar approach but the first file was loaded as this was used as a test 
    file_name_initial = "era5_2m_temperature_2021_01_01.nc"
    path_flash_initial = "/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/ERA5/Jan2021_2mTemp/"
    t_initial = path_flash_initial+file_name_initial
    combined_2m_temp = xr.open_dataset(t_initial)

    for i in range(2, 8):
        # loop combining the daily era5 2m_temperature data into one dataframe
        file_name_loop = "era5_2m_temperature_2021_01_0"+str(i)+".nc"
        path_flash = "/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/ERA5/Jan2021_2mTemp/"
        t = path_flash+file_name_loop

        dset_E5 = xr.open_dataset(t)
        combined_2m_temp = xr.concat([combined_2m_temp, dset_E5], dim="time")
    return combined_2m_temp
        
print('testing')
combined_2m_temp = xr.Dataset()
combined_2m_temp = com_temp_files(combined_2m_temp)
''''''
# Loop over each flash and extract the 2m temperature at its location and time

def combine(combined_flash, combined_2m_temp):
    for index, flash in combined_flash.iterrows():
        # Extract the latitude and longitude of each flash 
        lat = flash['Latitude']
        lon = flash['Longitude']

        # Find the nearest latitude and longitude in the ERA5 dataset
        era5_lat = combined_2m_temp.latitude.sel(latitude=lat, method='nearest').values
        era5_lon = combined_2m_temp.longitude.sel(longitude=lon, method='nearest').values

        # Extract the 2m temperature at the nearest location and time and store it in the flash data
        t2m = combined_2m_temp.t2m.sel(latitude=era5_lat, longitude=era5_lon, time=flash['TA19 Start Time'], method='nearest').values
        
        #Change from kelvin to celsius for easier readability. 
        celsius_t2m = t2m - 273.15
        combined_flash.at[index, 'T2m_Celsius'] = celsius_t2m

'''
#Modification to script so that this does not need to be run every time as it takes significant time for my laptop 
#to complete this.
combine(combined_flash,combined_2m_temp)
combined_flash.to_csv("Jan_temp_flash.csv",index = False)

'''
combined_flash = pandas.read_csv("Jan_temp_flash.csv")
combined_flash['hfc_test'] = hourly_flash_count['hours']

# extract the hour from Start Time
combined_flash['Start Time_temp'] = pandas.to_datetime(combined_flash['TA19 Start Time'])
combined_flash['Hour_temp'] = combined_flash['Start Time_temp'].dt.hour

# Graph 2 - time vs flash 2m_temperature 
#prints the average temperature occurence of flashes for each hour of the day.
hourly_temp = combined_flash.groupby('Hour_temp')['T2m_Celsius'].mean()
print(hourly_temp)

#This is the average temperature of flashes in each hour for the whole week.
hfc_temp = combined_flash.groupby('hfc_test')['T2m_Celsius'].mean().reset_index()
hfc_temp.columns = ['hfc_test', 'T2m_Celsius']
origin_date = pandas.to_datetime('2021-01-01 00:00:00', utc=True)
hfc_temp['date_time'] = origin_date + pandas.to_timedelta(hfc_temp['hfc_test'], unit='h')

mean = combined_flash['T2m_Celsius'].mean()
std_dev = combined_flash['T2m_Celsius'].std()

print(std_dev)
print('Mean flash surface temperature for Jan 01 - Jan 07 2021 = ')
print(mean)

#Plots on the graph the mean flash temperature and a comparison against mean when extremes are removed. 
ex_95 = combined_flash['T2m_Celsius'].quantile(0.95)
ex_5 = combined_flash['T2m_Celsius'].quantile(0.05)
mean_non_extreme = combined_flash[(combined_flash['T2m_Celsius'] < ex_95) & (combined_flash['T2m_Celsius'] > ex_5)]['T2m_Celsius'].mean()

print('90th Percentile mean flash surface temperature for Jan 01 - Jan 07 2021 = ')
print(mean_non_extreme)

plt.scatter(combined_flash['Start Time_temp'], combined_flash['T2m_Celsius'], s=2)
plt.xlabel('Time')
plt.ylabel('Temperature (Celsius)')
plt.title('Lightning flash relationship w/ 2m surface temperature  [01-01-2021 to 07-01-2021]')

# plot mean line
plt.axhline(y=mean, color='r', linestyle='-', label='Mean')
plt.axhline(y=mean_non_extreme, color='orange', linestyle='-', label='90 percentile mean')

plt.plot(hfc_temp['date_time'], hfc_temp['T2m_Celsius'], '-o', color='green', linewidth=0.5, markersize = 2,label='Hourly Avg Temp')
plt.legend()
plt.show()

#Third Graph flash count vs avg temp 
#hourly_bar = pandas.Series(hours_list).value_counts().sort_index()
g3_hour = hourly_bar.reset_index().rename(columns={'index': 'Hour', 0: 'Count'})
g3_hour['date_time'] = origin_date + pandas.to_timedelta(g3_hour['Hour'], unit='h')

fig, ax_count = plt.subplots()
ax_temp = ax_count.twinx()
ax_count.bar(g3_hour['date_time'], g3_hour['Count'],alpha = 0.5,width=pandas.Timedelta(hours=1))
ax_count.set_xlabel('Date')
ax_count.set_ylabel('Count', color='blue')

ax_temp.plot(hfc_temp['date_time'], hfc_temp['T2m_Celsius'], '-o', color='green', linewidth=0.5, markersize = 2,label='Hourly Avg Temp')
ax_temp.set_ylabel('Temperature (Celsius)', color='green')
ax_temp.legend()
plt.title('Flash count vs average 2m_temperature - Jan')
plt.show()


