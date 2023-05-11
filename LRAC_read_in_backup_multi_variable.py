from pickle import FALSE, TRUE
from tkinter.ttk import LabelFrame
import netCDF4
import numpy as np 
import matplotlib.pyplot as plt
import csv
import datetime
import pandas
import xarray as xr
import cartopy.crs as ccrs

import scipy
from scipy import stats

import ennemi 
from ennemi import estimate_entropy
from ennemi import estimate_corr
from ennemi import estimate_mi , normalize_mi

from sklearn.utils import resample

file_name = "LISOTD_LRAC_V2.3.2015.nc"
f2 =  "LISOTD_LRADC_V2.3.2015.nc"
path = "/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/LRAC/"
LRAC_file = path+file_name
LRADC_file = path +f2

'''
LRADC
LRADC_COM_SMFR
Annual cycle of diurnal cycle (UTC) of flash rate (Flashes km-2 day-1)
365 x 6 x 144 x 72
1 day x 4 hours (UTC) x 2.5° x 2.5°
7.5° x 7.5° boxcar moving average, 111-day boxcar moving average

LRADC_COM_SMFR2
'''

LRAC_dset = xr.open_dataset(LRAC_file)
LRADC_dset = xr.open_dataset(LRADC_file)
LRAC_netC = netCDF4.Dataset(LRAC_file)
LRADC_netC = netCDF4.Dataset(LRADC_file)

print('Variables available in the LRAC file:')
print(LRAC_netC.variables.keys())
print('/////////////////////////////////////////')
print('Variables available in the LRADC file:')
print(LRADC_netC.variables.keys())

def extract_v(file_Xarray, variable_name): 
    new_v = file_Xarray[variable_name]
    return new_v 

LRAC_COM_FR = extract_v(LRAC_dset,"LRAC_COM_FR")
LRAC_LIS_DVT = extract_v(LRAC_dset,"LRFC_LIS_DVT")
LRAC_OTD_DVT = extract_v(LRAC_dset,"LRFC_OTD_DVT")
LRAC_latitude = extract_v(LRAC_dset,"Latitude")
LRAC_longitude = extract_v(LRAC_dset,"Longitude")

print('Loaded_1')
LRADC_COM_SF = extract_v(LRADC_dset,"LRADC_COM_SF")
LRADC_COM_SMFR = extract_v(LRADC_dset,"LRADC_COM_SMFR")
LRADC_COM_SMFR2 = extract_v(LRADC_dset,"LRADC_COM_SMFR2")
print('Loaded_2')


def show_daily_graph(Direct_viewing_time): 
    y = np.arange(1, 25)
    x = np.arange(1, 366)
    X, Y = np.meshgrid(x, y)

    # Create a contour plot
    c = ax.contourf(Y, X, Direct_viewing_time, cmap='coolwarm')
    ax.set_xlabel('Local Hour')
    ax.set_ylabel('Day of Year')
    ax.set_title('LRAC_OTD_DVT Contour Plot')
    cbar = plt.colorbar(c)

    weeks = [1, 14, 27, 40]
    # Lines representing the high res investigation for JAN,APR,JUL and OCT
    for week in weeks:
        y_val = week * 7
        ax.axhline(y_val, color='black', linestyle='--')

    plt.show()

fig, ax = plt.subplots()
show_daily_graph(LRAC_OTD_DVT)

'''
End of diurnal investigation. 
'''

print(LRAC_COM_FR.dims)


def show_heatmap(LRAC_COM_FR):
    ax2 = fig2.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    LRAC_COM_FR[:,:,6].plot(ax=ax2, cmap='jet', transform=ccrs.PlateCarree())
    ax2.coastlines()
    plt.show()

def show_heatmap_2(LRAC_COM_FR):
    ax2 = fig2.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    ax2.imshow(LRAC_COM_FR[:,:,0], cmap='jet', transform=ccrs.PlateCarree(), origin='lower', extent=[-180, 180, -90, 90])
    ax2.coastlines()
    plt.show()



# List of month names
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# Initialize an empty array to hold temperature values
global_temps = np.array([])

d2m_temps = []
RH_temps = []
d2m_pattern_temps = []
RH_pattern_temps = []

# Loop through each month and read the corresponding CSV file
for month in months:
    print(month)
    csv_file = '/Users/admin/Documents/FInal_Semester_Project/Python/ERA5_script_output/{}_Global.csv'.format(month)
    a = 1
    with open(csv_file, 'r') as f:
        next(f)  # Skip row
        temps = np.array([float(row.split(',')[1]) for row in f])
        global_temps = np.append(global_temps, temps)

    #multi variable addition
    csv_file_hum = '/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/ERA5/Monthly_humidity/{}_Global_Humidity.csv'.format(month)
    with open(csv_file_hum, 'r') as f_1:
        #Date,d2m_mean,d2m_pattern,RH_mean,RH_pattern
        next(f_1)  # Skip row
        for i, row in enumerate(f_1):
            if i % 2 == 0:
                line = row.split(',')
                t1 = float(line[1])
                d2m_temps.append(t1)

                d = line[2]
                e = d.strip('"[')
                f = e.strip(']"\n')

                pattern_temps_1 = [float(x) for x in f.split()]
                d2m_pattern_temps.append(pattern_temps_1)

            else:
                line1 = row.split(',')
                t2 = float(line1[0])
                RH_temps.append(t2)

                d1 = line1[1]
                e1 = d1.strip('"[')
                f1 = e1.strip(']"\n')

                pattern_temps_2 = [float(x) for x in f1.split()]
                RH_pattern_temps.append(pattern_temps_2)
            
def new_var_read_in(months):
    var_temps = []
    var_patterns = []
    for month in months:
        print(month)
        csv_file = '/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/ERA5/monthly_mvimd/{}_mvimd.csv'.format(month)
        a = 1
        with open(csv_file, 'r') as f:
            next(f)  # Skip row
            for row in f:
                line = row.split(',')
                t1 = float(line[1])
                var_temps.append(t1)

                d1 = line[2]
                e1 = d1.strip('"[')
                f1 = e1.strip(']"\n')

                pattern_temps_2 = [float(x) for x in f1.split()]
                var_patterns.append(pattern_temps_2)
    return var_temps,var_patterns


mvimd_temps , mvimd_patterns = new_var_read_in(months)
#cape_temps , cape_patterns = new_var_read_in(months)
#k_temps , k_patterns = new_var_read_in(months)

#Stage 5 modification 
month_s5 = ['JAN_S5']
m1 = ['JAN_S5', 'FEB_S5'] 
m2 = ['MAR_S5', 'APR_S5', 'MAY_S5']
m3 = ['JUN_S5', 'JUL_S5', 'AUG_S5']
m4 = ['SEP_S5', 'OCT_S5', 'NOV_S5']
m5 = ['DEC_S5']

global_diurnal_pattern = []
pattern_temps = []
def month_diurnal(m):
    global_diurnal_pattern = []
    pattern_temps = []
    for mon in m:
        print(mon)
        csv_file_s5 = '/Users/admin/Documents/FInal_Semester_Project/Python/{}_Global.csv'.format(mon)
        with open(csv_file_s5, 'r') as f1:
            next(f1)  # Skip row
            for row in f1:
                c = row.split(',')
                d = c[1]
                e = d.strip('"[')
                f = e.strip(']"\n')
                
                pattern_temps = [float(x) for x in f.split()]
                global_diurnal_pattern.append(pattern_temps)
    return(global_diurnal_pattern)

global_diurnal_pattern = month_diurnal(month_s5)
pattern_1 = month_diurnal(m1)
pattern_2 = month_diurnal(m2)
pattern_3 = month_diurnal(m3)
pattern_4 = month_diurnal(m4)
pattern_5 = month_diurnal(m5)


a_0 = np.mean(pattern_1[:30], axis=0)
a_01 = np.mean(pattern_1[31:], axis=0)
a_02 = np.mean(pattern_5, axis=0)


a_1 = np.mean(pattern_2[:30], axis=0)
a_2 = np.mean(pattern_2[31:61], axis=0)
a_3 = np.mean(pattern_2[61:], axis=0)


a_4 = np.mean(pattern_3[:30], axis=0)
a_5 = np.mean(pattern_3[31:61], axis=0)
a_6 = np.mean(pattern_3[61:], axis=0)

a_7 = np.mean(pattern_4[:30], axis=0)
a_8 = np.mean(pattern_4[31:61], axis=0)
a_9 = np.mean(pattern_4[61:], axis=0)


winter = np.concatenate([pattern_1, pattern_5])
winter_temp = np.mean(winter, axis=0)

spring_temp = np.mean(pattern_2, axis=0)
summer_temp = np.mean(pattern_3, axis=0)
autumn_temp = np.mean(pattern_4, axis=0)


print(global_diurnal_pattern)




# Plot global temperature data
fig, ax1 = plt.subplots()

# Plot flash rate data
mean_flash_rate = np.mean(LRAC_COM_FR, axis=(0,1))
ax1.plot(np.arange(365), mean_flash_rate, linewidth=2, markersize=3, label='Mean Flash Rate')
ax1.set_xlabel('Day of the Year')
ax1.set_ylabel('Mean Flash Rate')
ax1.set_title('Global Diurnal Flash Rate vs Avg. daily temp')

# Create a secondary y-axis for temperature
ax2 = ax1.twinx()
ax2.plot(np.arange(365), global_temps, linewidth=2, markersize=3, label='Global Temperature',color = 'red')
ax2.set_ylabel('Avg. Daily Temperature (C)')


# Display legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
# Show the plot
plt.show()


#Created a function for mapping multi variable plotting 

def plot_annual(flash_var, second_var,label_v,unit):

    #fig, ax1 = plt.subplots()
# Plot flash rate data

    ax1.plot(np.arange(365), flash_var, linewidth=2, markersize=3, label='Mean Flash Rate')
    ax1.set_xlabel('Day of the Year')
    ax1.set_ylabel('Mean Flash Rate')
    ax1.set_title('Global Diurnal Flash Rate vs ' + label_v)

    # Create a secondary y-axis for temperature
    ax2 = ax1.twinx()
    ax2.plot(np.arange(365), second_var, linewidth=2, markersize=3, label=label_v,color = 'red')
    ax2.set_ylabel(label_v + unit)

    # Display legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # Show the plot
    plt.show()
    return 

fig, ax1 = plt.subplots()
plot_annual(mean_flash_rate ,mvimd_temps ,'kx','(J kg-1)')

'''
fig, ax1 = plt.subplots()
plot_annual(mean_flash_rate ,k_temps ,'kx','(J kg-1)')

fig, ax1 = plt.subplots()
plot_annual(mean_flash_rate ,cape_temps ,'CAPE ','(J kg-1)')
'''
fig, ax1 = plt.subplots()
plot_annual(mean_flash_rate ,d2m_temps ,'2m dewpoint temperature','(C)')

fig, ax1 = plt.subplots()
plot_annual(mean_flash_rate ,RH_temps ,'Calculated Relative Humidity','%')


def correlation_second_var(x,y,temp,labelx): 
    print("\\\\\\\\\\\\\\\\\\\\\\\\   Start of statistical analysis on strength of the relationship  /////////////////////////")
    print(labelx + "and Global Diurnal Flash Rate ")
    #print(f'Correlation Coefficient calculated with numpy function: {corr_coef:.4f}')
    print("")
    # Calculate Pearson's correlation coefficient for validation 

    corr_coef_1, p_value_1 = scipy.stats.pearsonr(x,y)
    print('Pearsons Correlation Coefficient: {:.4f}'.format(corr_coef_1))
    print('Corresponding p value: {:.4f}'.format(p_value_1))
    print('For testing purposes as p is very low here is the value:')
    print(p_value_1)

    print("")

    MI_cc = estimate_corr(x,y)
    print("")
    print(f"Mutual Information derived correlation coefficient = : {MI_cc[0,0]:.3f}")

    mi_condition = estimate_corr(x,y, cond=temp)
    print(f"Mutual Information with temperature already known = : {mi_condition[0,0]:.3f}")

    print("---------------------------------------------------------------------------------------")

    return 

correlation_second_var(mvimd_temps,mean_flash_rate,global_temps,'MVIMD_value')

'''
correlation_second_var(k_temps,mean_flash_rate,global_temps,'K_value')
correlation_second_var(cape_temps,mean_flash_rate,global_temps,'CAPE')

'''
correlation_second_var(d2m_temps,mean_flash_rate,global_temps,'2m_dewpoint_temperature')
correlation_second_var(RH_temps,mean_flash_rate,global_temps,'Relative Humidity ')











# Normalize the data
std_temp = (global_temps - np.mean(global_temps)) / np.std(global_temps)
std_flash_rate = (mean_flash_rate - np.mean(mean_flash_rate)) / np.std(mean_flash_rate)

def land_sea_mask(LRAC_latitude,LRAC_longitude): 
    file_name_LS = "lsm_1279l4_0.1x0.1.grb_v4_unpack.nc"
    path_LS = "/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/Land_sea_mask/"
    file_LS = path_LS+file_name_LS

    LS_mask_dset = xr.open_dataset(file_LS)
    LS_mask_netC = netCDF4.Dataset(LRADC_file)

    print(LRAC_latitude)
    print(LRAC_longitude)

    print('Variables available in the mask file:')
    print(LS_mask_netC.variables.keys())
    print('TEST:')
    print(LS_mask_dset)

    # find the nearest latitude and longitude value
    nearest_lat = LS_mask_dset['latitude'].sel(latitude=LRAC_latitude, method='nearest')
    nearest_lon = LS_mask_dset['longitude'].sel(longitude=LRAC_longitude + 180, method='nearest')

    # extract the corresponding mask value
    LS_mask_value = LS_mask_dset['lsm'].sel(latitude=nearest_lat, longitude=nearest_lon, method='nearest').values
    return LS_mask_value

LS_mask_value = land_sea_mask(LRAC_latitude,LRAC_longitude)

lfd_day = []
sfd_day = []

for day in range(LRAC_COM_FR.shape[2]):
    a = LS_mask_value[0,:,:]
    b = a.astype(bool)

    land_flash_rate_day = np.nanmean(np.where(b, LRAC_COM_FR[:, :, day], np.nan))
    sea_flash_rate_day = np.nanmean(np.where(~b, LRAC_COM_FR[:, :, day], np.nan))
    #sea_flash_rate_day = np.ma.average(np.ma.masked_array(LRAC_COM_FR[:, :, day], mask=a_bool), axis=(0, 1))

    lfd_day.append(land_flash_rate_day)
    sfd_day.append(sea_flash_rate_day)



fig, ax1 = plt.subplots()
ax1.plot(np.arange(365), lfd_day, linewidth=2, markersize=3, label='LAND Flash Rate')
ax1.plot(np.arange(365), sfd_day, linewidth=2, markersize=3, label='SEA Flash Rate')

ax1.set_xlabel('Day of the Year')
ax1.set_ylabel('Mean Flash Rate')
ax1.set_title('Land vs Sea Flash Rate')
ax1.legend(loc='upper left')
plt.show()


#Stage 3.1 Linear based assessment of the correlation between the two 
# Calculate the correlation coefficient
corr_coef = np.corrcoef(std_temp, std_flash_rate)[0, 1]
print("\\\\\\\\\\\\\\\\\\\\\\\\   Start of statistical analysis on strength of the relationship  /////////////////////////")
print("Global Diurnal Flash Rate and Mean Temperature Correlation")
#print(f'Correlation Coefficient calculated with numpy function: {corr_coef:.4f}')
print("")
# Calculate Pearson's correlation coefficient for validation 

corr_coef_1, p_value_1 = scipy.stats.pearsonr(std_temp, std_flash_rate)
print('Pearsons Correlation Coefficient: {:.4f}'.format(corr_coef_1))
print('Corresponding p value: {:.4f}'.format(p_value_1))
print('For testing purposes as p is very low here is the value:')
print(p_value_1)

print("")
'''
lfd_std_flash_rate = (lfd_day - np.mean(lfd_day)) / np.std(lfd_day)
corr_coef_2, p_value_2 = scipy.stats.pearsonr(std_temp, lfd_std_flash_rate)
print('LAND FR vs GLOBAL TEMP Pearsons Correlation Coefficient: {:.4f}'.format(corr_coef_2))
print('Corresponding p value: {:.4f}'.format(p_value_2))
print('For testing purposes as p is very low here is the value:')
print(p_value_2)
'''

plt.show()

#Stage 3.2 Correlation coefficient assessment assuming non linearity 
#First approach is Mutual Information analysis based upon entropy 
#Requires the use of the ennemi library : polsys.github.io/ennemi/what-is-entropy.html
# given that n = 365 a permutation test is better than p value from spearmans 



#print(dir(ennemi))

e1 = estimate_entropy(global_temps)
e1_std = estimate_entropy(std_temp)
print("Introduction of Mutual Information Approach: ")
#print(f"Entropy of avg. 2m surface temperature : {e1:.2f} nats")
#print(f"Entropy of avg. 2m surface temperature : {e1/np.log(2):.2f} bits")
print("")
print(f"Entropy of avg. 2m surface temperature w/std  : {e1_std:.2f} nats")
print(f"Entropy of avg. 2m surface temperature w/std  : {e1_std/np.log(2):.2f} bits")
print("")

e2 = estimate_entropy(mean_flash_rate)
e2_std = estimate_entropy(std_flash_rate)

#print(f"Entropy of avg. global flash rate : {e2:.2f} nats")
#print(f"Entropy of avg. global flash rate : {e2/np.log(2):.2f} bits")
#print("")
print(f"Entropy of avg. global flash rate w/std  : {e2_std:.2f} nats")
print(f"Entropy of avg. global flash rate w/std  : {e2_std/np.log(2):.2f} bits")

MI_cc = estimate_corr(global_temps,mean_flash_rate)
MI_cc_wstd = estimate_corr(std_temp,std_flash_rate)

print("")
print(f"Mutual Information derived correlation coefficient = : {MI_cc[0,0]:.3f}")
print(f"Mutual Information derived correlation coefficient w/ normalized dataset= : {MI_cc_wstd[0,0]:.3f}")


print("")
print("")
'''
#Can investigate the best nearest neighbour value - sensitivity test to show that method works consistently 
MI_a = estimate_mi(global_temps,mean_flash_rate,k = 3)
MI_wtsd = estimate_mi(std_temp,std_flash_rate , k = 3)

print("")
print(f"Mutual Information  = : {MI_a[0,0]:.2f}")
print(f"Mutual Information w/ normalized dataset  = : {MI_wtsd[0,0]:.2f}")

MI_a_norm = normalize_mi(MI_a[0,0])
MI_wtsd_norm = normalize_mi(MI_wtsd[0,0])

print("")
print(f"Normalized mutual Information  = : {MI_a_norm:.3f}")
print(f"Normalized mutual Information w/ normalized dataset  = : {MI_wtsd_norm:.3f}")
print("")
'''
#Start of the regression analysis - Calculation done first based on paper S1 
#Created within function to be used again for any calculated parameter 

def regression_single(x,y): 
    #Assume that the avg. temperature is x and the flash rate is y 
    n = 365 

    Ex = np.sum(x)
    Ey = np.sum(y)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x_squared = x * x 
    y_squared = y * y
    x_y = x * y 

    Ex_squared = np.sum(x_squared)
    Ey_squared = np.sum(y_squared)
    Exy = np.sum(x_y)

    Ex_squared_n = Ex_squared / n 
    Ey_squared_n = Ey_squared / n
    Exy_n = Exy / n 

    Qyx = Exy - ((1/n) * Ex * Ey)
    E_x_x_mean_squared = Ex_squared - ((Ex ** 2)/n)
    Qt = Ey_squared - ((Ey ** 2)/n)

    b = Qyx / E_x_x_mean_squared

    print('The regression line equation is: y = ' + str(y_mean.item()) + ' + '+ str(b.item()) + '(x - ' + str(x_mean) + ')')
    return  y_mean,b,x_mean

def calculate_r_squared(x, y, y_mean, b):
    y_predicted = y_mean + b * (x - np.mean(x))
    SSR = np.sum((y_predicted - y) ** 2)
    SST = np.sum((y - y_mean) ** 2)
    r_squared = 1 - (SSR / SST)
    return r_squared

y_mean,b,x_mean = regression_single(global_temps,mean_flash_rate)
print (y_mean.item())
y_offset = y_mean.item()
print(b.item())
gradient = b.item()
print(x_mean.item())
x_offset = x_mean.item()


fig, ax1 = plt.subplots()
ax1.scatter(global_temps,mean_flash_rate, s = 2.5, c = "Orange" ,label='Temp vs Flash rate')
ax1.set_xlabel('Avg 2m Surface Temperature')
ax1.set_ylabel('Mean Flash Rate')
reg_line = y_offset + gradient * (global_temps - x_offset)
ax1.plot(global_temps , reg_line , linewidth=1.5, markersize=3, label='Single regression Calculation')
plt.show()

'''
r_squared = calculate_r_squared(global_temps, mean_flash_rate, y_mean, b)
print(f'R-squared value: {r_squared:.4f}')
'''

print(LRADC_COM_SMFR2.dims)
diurnal_mean = LRADC_COM_SMFR2.mean(dim=('Latitude', 'Longitude', 'Day_of_Year'))
month_names = ['JFM', 'AMJ', 'JAS', 'OND']
date_range = pandas.date_range(start='2014-01-01', periods=364, freq='D')

fig, ax = plt.subplots(figsize=(10, 6))
ax2 = ax.twinx()
count = 0 
graph_counter = 0 
plot_g = FALSE

season_pattern_temp = []


for day in date_range:
    # Extract the diurnal pattern for the current day
    diurnal_pattern = LRADC_COM_SMFR2.sel(Day_of_Year=day,method='nearest').mean(dim=('Latitude', 'Longitude'))
    if count < 58: 
        c= "red"
        l = "DJF"
        #ax2.plot(np.arange(0, 24, 3), pattern_1[graph_counter], linewidth=2, markersize=2, alpha = 0.5, label='Global Temperature',color = 'gray')
    elif count < (58+92):
        c= "#02247A"
        l = "MAM"
        plot_g = FALSE
        #ax2.plot(np.arange(0, 24, 3), pattern_2[graph_counter - (58)], linewidth=2, markersize=2, alpha = 0.5, label='Global Temperature',color = 'gray')
    elif count < (58+92+92):
        c= "green"
        l = "JJA"
        plot_g = FALSE
        #ax2.plot(np.arange(0, 24, 3), pattern_3[graph_counter - (58+92)], linewidth=2, markersize=2, alpha = 0.5, label='Global Temperature',color = 'gray')
    elif count < (58+92+92+91):
        c= "orange"
        l = "SON"
        plot_g = FALSE
        #ax2.plot(np.arange(0, 24, 3), pattern_4[graph_counter - (58+92+92)], linewidth=2, markersize=2, alpha = 0.5, label='Global Temperature',color = 'gray')
    else:
        plot_g = FALSE
        c= "red"
        l = "DJF"
        print(graph_counter - (58+92+92+91))
        #ax2.plot(np.arange(0, 24, 3), pattern_5[graph_counter - (58+92+92+91)], linewidth=2, markersize=2, alpha = 0.5, label='Global Temperature',color = 'gray')

    if plot_g == TRUE:
        ax.plot(np.arange(0, 24, 2), diurnal_pattern, color = c, alpha = 0.3,label = l)
        season_pattern_temp.append(diurnal_pattern.values)
    #ax2.plot(np.arange(0, 24, 3), global_diurnal_pattern[graph_counter], linewidth=2, markersize=2, alpha = 0.5, label='Global Temperature',color = 'gray')
    
    count += 1
    graph_counter += 1   

ax2.plot(np.arange(0, 24, 3), winter_temp,'-.', linewidth=4, markersize=2, alpha = 1, label='Global Temperature',color = '#225BE9')

ax2.plot(np.arange(0, 24, 3), a_0, linewidth=3, markersize=2, alpha = 0.7, label='Global Temperature',color = '#22DEE9')
ax2.plot(np.arange(0, 24, 3), a_01, linewidth=3, markersize=2, alpha = 0.7, label='Global Temperature',color = '#22DEE9')
ax2.plot(np.arange(0, 24, 3), a_02, linewidth=3, markersize=2, alpha = 0.7, label='Global Temperature',color = '#22DEE9')


ax.plot(np.arange(0, 24, 2), diurnal_mean,'--', color = 'black', alpha = 1,linewidth=4,label = 'annual average')
ax.set_title('Autumn Diurnal Lightning Flash Rate {SON flash rate[Orange] global mean [Black]} vs 2m temperature {Seasonal (Blue) monthly(Light blue)}')
ax.set_xlabel('Hour of the Day (UTC)')
ax.set_ylabel('Flash Rate (Flashes km$^{-2}$ day$^{-1}$)')
ax2.set_ylabel('Avg. 2m Temperature (C)')
ax.grid()
plt.show()
print("Start of PART 2")


fig, ax = plt.subplots(figsize=(10, 6))
# Define latitude ranges for tropics and non-tropics regions
tropics_lat_range = slice(-23.5, 23.5)
extended_tropics_lat_range = slice(-30, 30)
# Define non-tropics latitude range as all values outside of tropics latitude range
non_tropics_lat_range_1 = slice(None, -23.5)  # values less than -23.5
non_tropics_lat_range_2 = slice(23.5, None)  # values greater than 23.5

print(non_tropics_lat_range_1)

# Extract LRADC_COM_SMFR2 values for the tropics and non-tropics regions
tropics_data = LRADC_COM_SMFR2.sel(Latitude=tropics_lat_range)
t_d_diurnal = tropics_data.mean(dim=('Latitude', 'Longitude', 'Day_of_Year')).values
extended_tropics_data = LRADC_COM_SMFR2.sel(Latitude=extended_tropics_lat_range)
et_d_diurnal = extended_tropics_data.mean(dim=('Latitude', 'Longitude', 'Day_of_Year')).values
non_tropics_data = xr.concat(
    [LRADC_COM_SMFR2.sel(Latitude=non_tropics_lat_range_1), 
     LRADC_COM_SMFR2.sel(Latitude=non_tropics_lat_range_2)], dim='Latitude')
nt_d_diurnal = non_tropics_data.mean(dim=('Latitude', 'Longitude', 'Day_of_Year')).values

tropics_data_shape = tropics_data.shape
extended_tropics_data_shape = extended_tropics_data.shape
non_tropics_data_shape = non_tropics_data.shape

print("Tropics data shape:", tropics_data_shape)
print("Extended tropics data shape:", extended_tropics_data_shape)
print("Non-tropics data shape:", non_tropics_data_shape)

ax.plot(np.arange(0, 24, 2), t_d_diurnal, color = 'red', alpha = 0.7)
ax.plot(np.arange(0, 24, 2), et_d_diurnal , color = 'green', alpha = 0.7)
ax.plot(np.arange(0, 24, 2), nt_d_diurnal , color = 'blue', alpha = 0.7)
ax.plot(np.arange(0, 24, 2), diurnal_mean, color = 'black', alpha = 1,)

ax.set_title('Diurnal Lightning Flash Rate [Tropics = Red, Non-Tropics = Blue]')
ax.set_xlabel('Hour of the Day (UTC)')
ax.set_ylabel('Flash Rate (Flashes km$^{-2}$ day$^{-1}$)')
ax.grid()
plt.show()


#Start of estimating correlation for diurnal seasonal patterns 
#Decided to linearly interpolate and bootstrap datasets to get a better input for MI based correlation estimate. 



#Only doing winter to see if the result warrants any extension 
#winter_temp will be used alongside the flash rates that go with DJF.
# 
# 


'''

winter_flash_single = season_pattern_temp
winter_flash = np.mean(winter_flash_single, axis=0)


#Will be used as a test to make sure it works. 
test_no = 48
print("Test for flash interpolation: ")
interp_spacing = np.linspace(0, len(winter_temp) - 1, num = test_no)
interp_spacing_2 = np.linspace(0, len(winter_flash) - 1, num = test_no)

interpolated_temp = np.interp(interp_spacing, np.arange(len(winter_temp)), winter_temp)
interpolated_flash_rate = np.interp(interp_spacing_2, np.arange(len(winter_flash)), winter_flash)
print("")
print(interpolated_flash_rate)

bootstrapped_temp_1 = resample(interpolated_temp, replace=True, n_samples=48, random_state=7)
bootstrapped_temp_2 = resample(interpolated_temp, replace=False, n_samples=48, random_state=77)
bootstrapped_temp_3 = resample(interpolated_temp, replace=False, n_samples=20, random_state=52)


fig, ax_new = plt.subplots()
ax_2_new = ax_new.twinx()
x_axis_int = np.linspace(0, 23, test_no)
test_int = np.linspace(0, 23, 20)
ax_new.plot(x_axis_int, interpolated_temp, linewidth=2, markersize=3, label='interp_winter_temp')
ax_2_new.plot(x_axis_int, interpolated_flash_rate, linewidth=2, markersize=3, color = 'red', label='interp_winter_flash')


ax_new.plot(x_axis_int, bootstrapped_temp_1 , '--',alpha = 0.3, linewidth=1, markersize=3, label='bootstrap_1')
ax_new.plot(x_axis_int, bootstrapped_temp_2 ,'--',alpha = 0.3, linewidth=1, markersize=3, label='bootstrap_2')
ax_new.plot(test_int, bootstrapped_temp_3 ,'--',alpha = 0.3,  linewidth=1, markersize=3, label='bootstrap_3')

ax_new.set_xlabel('Day of the Year')
ax_new.set_ylabel('Avg Temperature')
ax_2_new.set_ylabel('Mean Flash Rate')
ax_new.set_title('Extension of data_sets [WINTER]')
ax_new.legend(loc='upper left')
ax_2_new.legend(loc='upper right')
plt.show()



print("")
print('Test of two interpolated datasets [ WINTER ]')
corr_coef_2, p_value_2 = scipy.stats.pearsonr(interpolated_flash_rate,interpolated_temp,)
print('Pearsons Correlation Coefficient: {:.4f}'.format(corr_coef_2))
print('Corresponding p value: {:.4f}'.format(p_value_2))
print('For testing purposes as p is very low here is the value:')
print(p_value_2)

print("")
print('Now using MI method introducing a time delay investigation: ')

#As 100 points have been taken the time shift correlates to 14 .4 minute intervals 

time_lags = np.arange(0, 16)

mi = estimate_corr(interpolated_flash_rate,interpolated_temp,
                   lag=time_lags)

print(mi)

plt.plot(time_lags, mi, label="Correlation_Coefficient", marker="o")
plt.legend()
plt.xlabel("Time lag (steps)")
plt.ylabel("MI correlation")
plt.title("Mutual information between flash rate and temperature")
plt.show() # or plt.savefig(...)

print('end of script')
'''

def graph_stage_5a(x,y,season,lab): 
    season_flash_single = y
    season_flash = np.mean(season_flash_single, axis=0)
    test_no = 48
    print("Test for flash interpolation: ")
    interp_spacing = np.linspace(0, len(x) - 1, num = test_no)
    interp_spacing_2 = np.linspace(0, len(season_flash) - 1, num = test_no)

    interpolated_temp = np.interp(interp_spacing, np.arange(len(x)), x)
    interpolated_flash_rate = np.interp(interp_spacing_2, np.arange(len(season_flash)), season_flash)
    fig, ax_new = plt.subplots()
    ax_2_new = ax_new.twinx()
    x_axis_int = np.linspace(0, 23, test_no)
    test_int = np.linspace(0, 23, 20)
    ax_new.plot(x_axis_int, interpolated_temp, linewidth=2, markersize=3, label='interp_' +season+'_'+lab)
    ax_2_new.plot(x_axis_int, interpolated_flash_rate, linewidth=2, markersize=3, color = 'red', label='interp_'+season+'_flash')
    ax_new.set_xlabel('Time of Day (UTC)')
    ax_new.set_ylabel(lab)
    ax_2_new.set_ylabel('Mean Flash Rate')
    ax_new.set_title('Extension of data_sets [' + season + ']')
    ax_new.legend(loc='upper left')
    ax_2_new.legend(loc='upper right')
    return(interpolated_temp,interpolated_flash_rate)

def Correlation_S5(x,y,lab): 
    corr_coef_2, p_value_2 = scipy.stats.pearsonr(y,x)
    print('Pearsons Correlation Coefficient: {:.4f}'.format(corr_coef_2))
    print('Corresponding p value: {:.4f}'.format(p_value_2))
    print('For testing purposes as p is very low here is the value:')
    print(p_value_2)

    print("")
    print('Now using MI method introducing a time delay investigation: ')

    #As 100 points have been taken the time shift correlates to 14 .4 minute intervals 

    time_lags = np.arange(0, 16)
    mi = estimate_corr(y,x, lag=time_lags)
    print(mi)
    x_axis_lags = np.arange(0, 8, 0.5)
    plt.plot(x_axis_lags, mi, label="Correlation_Coefficient", marker="o")
    plt.legend()
    plt.xlabel("Time lag (hrs)")
    plt.ylabel("MI correlation")
    plt.title("Mutual information between flash rate and " + lab)
    return(mi)

'''
fig, ax_new = plt.subplots()
interpolated_temp,interpolated_flash_rate = graph_stage_5a(autumn_temp,season_pattern_temp,'AUTUMN','Avg Temperature')
plt.show()
mi = Correlation_S5(interpolated_temp,interpolated_flash_rate,'Avg Temperature')
plt.show()
'''



def switch_to_seasonal(pattern): 
    temp = pattern[:58] + pattern[-31:]

    winter = np.mean(temp,axis = 0)
    spring = np.mean(pattern[58:149],axis = 0)
    summer = np.mean(pattern[150:241],axis = 0)
    autumn = np.mean(pattern[242:333],axis = 0)

    return winter,spring,summer,autumn 

print('end of script')



#
#a_7 = np.mean(pattern_4[:30], axis=0)
#
w1,sp1,su1,a1 = switch_to_seasonal(d2m_pattern_temps)
w2,sp2,su2,a2 = switch_to_seasonal(RH_pattern_temps)

w5,sp5,su5,a5 = switch_to_seasonal(mvimd_patterns)
'''
w3,sp3,su3,a3 = switch_to_seasonal(cape_patterns)
w4,sp4,su4,a4 = switch_to_seasonal(k_patterns)
'''

fig, ax_new = plt.subplots()
interpolated_mvimd,interpolated_flash_rate = graph_stage_5a(a5,season_pattern_temp,'AUTUMN','mvimd')
plt.show()
mi_kx = Correlation_S5(interpolated_mvimd,interpolated_flash_rate,'mvimd_value')
plt.show()

'''
fig, ax_new = plt.subplots()
interpolated_cape,interpolated_flash_rate = graph_stage_5a(a3,season_pattern_temp,'AUTUMN','CAPE')
plt.show()
mi_rh = Correlation_S5(interpolated_cape,interpolated_flash_rate,'CAPE')
plt.show()


fig, ax_new = plt.subplots()
interpolated_dp,interpolated_flash_rate = graph_stage_5a(sp1,season_pattern_temp,'SPRING','2m Dewpoint Temp')
plt.show()
mi_dp = Correlation_S5(interpolated_dp,interpolated_flash_rate,'2m Dewpoint Temp')
plt.show()

fig, ax_new = plt.subplots()
interpolated_rh,interpolated_flash_rate = graph_stage_5a(sp2,season_pattern_temp,'SPRING','Relative Humidity')
plt.show()
mi_rh = Correlation_S5(interpolated_rh,interpolated_flash_rate,'Relative Humidity')
plt.show()
'''
print('end of script')