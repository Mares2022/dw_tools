# %%
import os
import dfm_tools as dfmt
import pandas as pd
import hydrolib.core.dflowfm as hcdfm

# %%
# model
lon_min, lon_max, lat_min, lat_max = -17, 15, 41, 66

dir_base = r'C:\Ocean\Work\Projects\2025\Schelde\Data' #This needs to be updated
dir_output = os.path.join(dir_base, f'CMEMS') 
overwrite = True # used for downloading of forcing data. Always set to True when changing the domain

#dates as understood by pandas.period_range(). ERA5 has freq='M' (month) and CMEMS has freq='D' (day)
date_min = '2018-01-01' # for the .bc files
date_max = '2019-01-01'  
ref_date = '2017-12-01'

# %%
# Model
model_name = 'model'
path_style = 'unix' # windows / unix

dir_output_bc =  os.path.join(dir_base, f'sea_boundary')
# poly_file = r'C:\Ocean\Work\Projects\2025\Schelde\Data\sea_boundary\SW_4326.pli'
# poly_file = r'C:\Ocean\Work\Projects\2025\Schelde\Data\sea_boundary\NE_4326.pli'
poly_file = r'C:\Ocean\Work\Projects\2025\Schelde\Data\sea_boundary\NW_4326.pli'

# %%
# Split the full period into months
# Create a date range with a monthly frequency
date_range = pd.date_range(start=date_min, end=date_max, freq='MS')
# Create start and end times list
start_times = date_range[:-1]  # exclude the last element for start times
end_times = date_range[1:] - pd.Timedelta(days=1)  # take the day before the next month's start as end time
# Combine start and end times into a list of tuples
monthly_periods = [(str(start), str(end)) for start, end in zip(start_times, end_times)]
# Handle the case for the final period (from start of the last month to date_max)
monthly_periods.append((str(date_range[-1]), date_max))
# Print the result
for period in monthly_periods:
    print(f"Start: {period[0]}, End: {period[1]}")

# %%
# CMEMS - download
os.makedirs(dir_output, exist_ok=True)
for period in monthly_periods:
    #You will need to download monthly no3, po4, si, o2 and phyc to create the variables you need for the WQ model.
    for varkey in ['o2','no3','po4','si','phyc','thetao']:
        # dfmt.download_CMEMS(varkey=varkey, freq='M',
        #                     longitude_min=lon_min, longitude_max=lon_max, latitude_min=lat_min, latitude_max=lat_max,
        #                     date_min=date_min, date_max=date_max,
        #                     dir_output=f'{dir_output}_monthly', file_prefix='cmems_', overwrite=overwrite,
        #                     dataset_id='cmems_mod_glo_phy_my_0.083deg_P1M-m')       # physics
        #                     # dataset_id = 'cmems_mod_glo_bgc_my_0.25deg_P1M-m')    # biogeochem                                                     # physics
        dfmt.download_CMEMS(varkey=varkey, freq='M',
                            longitude_min=lon_min, longitude_max=lon_max, latitude_min=lat_min, latitude_max=lat_max,
                            date_min=period[0], date_max=period[1],
                            dir_output=f'{dir_output}', file_prefix='cmems_', overwrite=overwrite) 
                            # dataset_id='cmems_mod_glo_phy_my_0.083deg_P1M-m')      # physics
                            # dataset_id='cmems_mod_glo_bgc_my_0.25deg_P1M-m')       # biogeochem 

# %%
# generate new format external forcings file (*.ext): initial and open boundary condition
ext_file_new = os.path.join(dir_output_bc, f'{model_name}_new.ext')
ext_new = hcdfm.ExtModel()

# %%
## Update the conversion dict:
conversion_dict = { # conversion is phyc in mmol/m3 to newvar in g/m3
                    'tracerbndOXY'        : {'ncvarname': 'o2',          'unit': 'g/m3', 'conversion': 32./1000.},
                    'tracerbndNO3'        : {'ncvarname': 'no3',         'unit': 'g/m3', 'conversion': 14./1000.},
                    'tracerbndPO4'        : {'ncvarname': 'po4',         'unit': 'g/m3', 'conversion': 30.97/1000.},
                    'tracerbndSi'         : {'ncvarname': 'si',          'unit': 'g/m3', 'conversion': 28.08/1000.},
                    'tracerbndPON1'       : {'ncvarname': 'phyc',        'unit': 'g/m3', 'conversion': 2. * (16./106.) * (14./1000.)}, # Caution: this empirical relation might not be applicable to your use case
                    'tracerbndPOP1'       : {'ncvarname': 'phyc',        'unit': 'g/m3', 'conversion': 2. * (1./106.) * (30.97/1000.)}, # Caution: this empirical relation might not be applicable to your use case
                    'tracerbndPOC1'       : {'ncvarname': 'phyc',        'unit': 'g/m3', 'conversion': 2. * (12./1000.)}, # Caution: this empirical relation might not be applicable to your use case
                    # 'tracerbndDON'        : {'ncvarname': 'phyc',        'unit': 'g/m3', 'conversion': (19./225.) * (91.8 / 8.2) * 2. * (14./1000.)}, # Caution: this empirical relation might not be applicable to your use case
                    # 'tracerbndDOP'        : {'ncvarname': 'phyc',        'unit': 'g/m3', 'conversion': (1./225.) * (91.8 / 8.2) * 2. * (30.97/1000.)}, # Caution: this empirical relation might not be applicable to your use case
                    'tracerbndDOC'        : {'ncvarname': 'phyc',        'unit': 'g/m3', 'conversion': (91.8 / 8.2) * 2. * (12./1000.)}, # Caution: this empirical relation might not be applicable to your use case
                    'tracerbndOpal'       : {'ncvarname': 'phyc',        'unit': 'g/m3', 'conversion': 0.5 * 0.13 * 28.08/1000}, # Caution: this empirical relation might not be applicable to your use case
                    # 'tracerbndTIC'        : {'ncvarname': 'tic',         'unit': 'g/m3', 'conversion': 1.}, 
                    # 'tracerbndALKA'       : {'ncvarname': 'alka',        'unit': 'g/m3', 'conversion': 1.}, 
                    # 'salinitybnd'         : {'ncvarname': 'so'},          #'1e-3'
                    'temperaturebnd'      : {'ncvarname': 'thetao'},      #'degC'
                    # 'ux'                  : {'ncvarname': 'uo'},          #'m/s'
                    # 'uy'                  : {'ncvarname': 'vo'},          #'m/s'
                    # 'waterlevelbnd'       : {'ncvarname': 'zos'},         #'m' #steric
                    # 'tide'                : {'ncvarname': ''},            #'m' #tide (dummy entry)
                    }


# %%
list_quantities = ['tracerbndOXY','tracerbndNO3','tracerbndPO4','tracerbndSi', 'tracerbndPON1', 'tracerbndPOP1',
                   'tracerbndPOC1', 'tracerbndDOC', 'tracerbndOpal','temperaturebnd'
                   ]

dir_pattern = os.path.join(dir_output, 'cmems_{ncvarname}_*.nc')

ext_new = dfmt.cmems_nc_to_bc(ext_new=ext_new,
                              refdate_str=f'minutes since {ref_date} 00:00:00 +00:00',
                              dir_output=dir_output_bc,
                              list_quantities=list_quantities,
                              tstart=date_min,
                              tstop=date_max, 
                              file_pli=poly_file,
                              dir_pattern=dir_pattern)
