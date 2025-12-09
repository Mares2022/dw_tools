import os
import matplotlib.pyplot as plt

import pandas as pd
import dfm_tools as dfmt
import hydrolib.core.dflowfm as hcdfm

# model
model_name, lon_min, lon_max, lat_min, lat_max = 'nz_monthly', -16, 14, 42, 65

# dates as understood by pandas.period_range(). ERA5 has freq='M' (month) and CMEMS has freq='D' (day)
date_min = '2011-12-01'
date_max = '2021-02-01'  
ref_date = '2011-12-22'

dir_base = r'p:\11209731-002-ospar-nutrients\data'

dir_output = os.path.join(dir_base, r'3-model_inputs\boundaries', f'CMEMS_{model_name}')
dir_output_data_cmems = os.path.join(dir_base, fr'1-external\CMEMS_{model_name}')

# poly_file = os.path.join(dir_base, fr'model_inputs\boundaries\extra_rand_dcsm.pli')         # not sure if and where use these boundaries...
poly_file = os.path.join(dir_base, fr'3-model_inputs\boundaries\DCSM-FM_OB_all_20181108.pli')

## Update the conversion dict:
conversion_dict = { # conversion is phyc in mmol/m3 to newvar in g/m3
                    'tracerbndOXY'        : {'ncvarname': 'o2',          'unit': 'g/m3', 'conversion': 32./1000.},
                    'tracerbndNO3'        : {'ncvarname': 'no3',         'unit': 'g/m3', 'conversion': 14./1000.},
                    'tracerbndPO4'        : {'ncvarname': 'po4',         'unit': 'g/m3', 'conversion': 30.97/1000.},
                    'tracerbndSi'         : {'ncvarname': 'si',          'unit': 'g/m3', 'conversion': 28.08/1000.},
                    'tracerbndPON1'       : {'ncvarname': 'phyc',        'unit': 'g/m3', 'conversion': 2. * (16./106.) * (14./1000.)}, # Caution: this empirical relation might not be applicable to your use case
                    'tracerbndPOP1'       : {'ncvarname': 'phyc',        'unit': 'g/m3', 'conversion': 2. * (1./106.) * (30.97/1000.)}, # Caution: this empirical relation might not be applicable to your use case
                    'tracerbndPOC1'       : {'ncvarname': 'phyc',        'unit': 'g/m3', 'conversion': 2. * (12./1000.)}, # Caution: this empirical relation might not be applicable to your use case
                    'tracerbndDON'        : {'ncvarname': 'phyc',        'unit': 'g/m3', 'conversion': (19./225.) * (91.8 / 8.2) * 2. * (14./1000.)}, # Caution: this empirical relation might not be applicable to your use case
                    'tracerbndDOP'        : {'ncvarname': 'phyc',        'unit': 'g/m3', 'conversion': (1./225.) * (91.8 / 8.2) * 2. * (30.97/1000.)}, # Caution: this empirical relation might not be applicable to your use case
                    'tracerbndDOC'        : {'ncvarname': 'phyc',        'unit': 'g/m3', 'conversion': (91.8 / 8.2) * 2. * (12./1000.)}, # Caution: this empirical relation might not be applicable to your use case
                    'tracerbndOpal'       : {'ncvarname': 'phyc',        'unit': 'g/m3', 'conversion': 0.5 * 0.13 * 28.08/1000}, # Caution: this empirical relation might not be applicable to your use case
                    'tracerbndTIC'        : {'ncvarname': 'tic',         'unit': 'g/m3', 'conversion': 1.}, 
                    'tracerbndAlka'       : {'ncvarname': 'alka',        'unit': 'g/m3', 'conversion': 1.}, 
                    'salinitybnd'         : {'ncvarname': 'so'},          #'1e-3'
                    'temperaturebnd'      : {'ncvarname': 'thetao'},      #'degC'
                    'ux'                  : {'ncvarname': 'uo'},          #'m/s'
                    'uy'                  : {'ncvarname': 'vo'},          #'m/s'
                    'waterlevelbnd'       : {'ncvarname': 'zos'},         #'m' #steric
                    'tide'                : {'ncvarname': ''},            #'m' #tide (dummy entry)
                    }

# Alternatively, use conversion_dict from dfmt  - Note: DON and DOP values different to the above.
# conversion_dict = dfmt.get_conversion_dict() 


## Function copied from dfm_tools --- so can change the conversion_dict, otherwise not (yet) specified input var:
def cmems_nc_to_bc(ext_bnd, list_quantities, tstart, tstop, file_pli, dir_pattern, dir_output, refdate_str=None):
    # input examples in https://github.com/Deltares/dfm_tools/blob/main/tests/examples/preprocess_interpolate_nc_to_bc.py
    
    file_bc_basename = os.path.basename(file_pli).replace('.pli','')
    for quantity in list_quantities:
        print(f'processing quantity: {quantity}')
        
        # times in cmems API are at midnight, so round to nearest outer midnight datetime
        tstart = pd.Timestamp(tstart).floor('1d')
        tstop = pd.Timestamp(tstop).ceil('1d')
        
        #open regulargridDataset and do some basic stuff (time selection, renaming depth/lat/lon/varname, converting units, etc)
        data_xr_vars = dfmt.open_dataset_extra(dir_pattern=dir_pattern, quantity=quantity,
                                               tstart=tstart, tstop=tstop,
                                               conversion_dict=conversion_dict,    
                                               refdate_str=refdate_str)
        # interpolate regulargridDataset to plipointsDataset
        data_interp = dfmt.interp_regularnc_to_plipoints(data_xr_reg=data_xr_vars, file_pli=file_pli)
        
        # convert plipointsDataset to hydrolib ForcingModel
        ForcingModel_object = dfmt.plipointsDataset_to_ForcingModel(plipointsDataset=data_interp)
        
        file_bc_out = os.path.join(dir_output,f'{quantity}_{file_bc_basename}_CMEMS.bc')
        
        ForcingModel_object.save(filepath=file_bc_out)
        
        # generate boundary object for the ext file (quantity, pli-filename, bc-filename)
        boundary_object = hcdfm.Boundary(quantity=quantity,
                                         locationfile=file_pli,
                                         forcingfile=ForcingModel_object)
        ext_bnd.boundary.append(boundary_object)
    
    return ext_bnd

ext_new = hcdfm.ExtModel()
dir_pattern = os.path.join(dir_output_data_cmems,'cmems_{ncvarname}_*.nc')

list_quantities = ['tracerbndNO3', 'tracerbndPO4', 'tracerbndSi', 'tracerbndPON1', 'tracerbndPOC1', 'tracerbndPOP1', 'tracerbndOpal', 'tracerbndDOC', 'tracerbndDON', 'tracerbndDOP']

cmems_nc_to_bc(ext_bnd=ext_new,
              refdate_str=f'minutes since {ref_date} 00:00:00 +00:00',
              dir_output=dir_output,
              list_quantities=list_quantities,
              tstart=date_min,
              tstop=date_max, 
              file_pli=poly_file,
              dir_pattern=dir_pattern)