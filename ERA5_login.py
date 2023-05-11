
import os
import cdsapi
import netCDF4

import urllib3
import certifi
import pandas


# Set up CDS API credentials
# Once this is done once this can be commented out. Included for understanding. 
cdsapirc_contents = """url: https://cds.climate.copernicus.eu/api/v2
key: 181245:d210f052-4217-4906-bbad-e23d52116c04
verify: 0"""
cdsapirc_dir = "/Users/admin/Documents/FInal_Semester_Project/Python/Login"
cdsapirc_path = os.path.join(cdsapirc_dir, ".cdsapirc")
with open(cdsapirc_path, "w") as f:
    f.write(cdsapirc_contents)

# Create a pool manager with certificate verification
http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())

# Use the CDS API to retrieve data
cds = cdsapi.Client(url='https://cds.climate.copernicus.eu/api/v2', verify=http)
# This is a test to see if the API data retrieval works. Implemented at the beginning but will be removed once testing verifies 
# correct working of the API
'''
Reference - https://cds.climate.copernicus.eu/api-how-to
'''

output_file = "ERA5_Test_Data.nc"
cds.retrieve("reanalysis-era5-pressure-levels", {
   "variable": "temperature",
    "pressure_level": "1000",
    "product_type": "reanalysis",
    "year": "2021",
    "month": "01",
    "day": "01",
    "time": "12:00",
    "format": "netcdf"
}, output_file)

print('Test A.1 - CDS API successfully retrieved test file ERA5_Test_Data.nc')


os.chdir('/Users/admin/Documents/FInal_Semester_Project/Python/Data_Disk/ERA5/Oct2021_2mTemp/')
def retrieve_func(day, month, year,parameter,filename):
    cds.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type':'reanalysis',
            'format':'netcdf',
            'variable':[
              parameter
            ],
            'year':[
                  year
            ],
            'month':[
                  month
            ],
            'day':[
                day
            ],
            'time':[
                '00:00','01:00','02:00',
                '03:00','04:00','05:00',
                '06:00','07:00','08:00',
                '09:00','10:00','11:00',
                '12:00','13:00','14:00',
                '15:00','16:00','17:00',
                '18:00','19:00','20:00',
                '21:00','22:00','23:00'
            ],
            'area':'90/-180/-90/179.75',
        },filename)


def getDataList(parameter):
    filename_list = [] 
    startDate = '2021-10-01'
    endDate = '2021-10-07'
    dataList = pandas.date_range(startDate,endDate).tolist()
    df = pandas.to_datetime(dataList)
    for i in df:
        print(i.year,i.month,i.day)
        filename = "era5_"+parameter+"_"+str(i.year)+"_"+str(i.month).zfill(2)+"_"+str(i.day).zfill(2)+".nc"
        retrieve_func(str(i.day).zfill(2), str(i.month).zfill(2), i.year, parameter,filename)
        filename_list.append(filename)

    return filename_list 


filelist_2m = getDataList('2m_temperature')
print('Completed the function - Test A complete')
