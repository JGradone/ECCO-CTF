# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
sys.path.append('/Users/joegradone/SynologyDrive/Drive/Rutgers/Research/code/GitHub/ECCOv4-py')
import ecco_v4_py as ecco
#from ecco_download import *
import numpy as np
from os.path import join,expanduser
import xarray as xr
from getpass import getpass
from http.cookiejar import CookieJar
from pathlib import Path
from netrc import netrc
import os
import re
# library to download files
from urllib import request
import glob
import xgcm
from xgcm import Grid
from dask.diagnostics import ProgressBar
user_home_dir = expanduser('~')
_netrc = join(user_home_dir)


################################################################################################
##                         These are all from the ECCO python tutorial                         #
################################################################################################
# not pretty but it works
def setup_earthdata_login_auth(url: str='urs.earthdata.nasa.gov'):
    # look for the netrc file and use the login/password
    try:
        username, _, password = netrc(file=_netrc).authenticators(url)
    
    # if the file is not found, prompt the user for the login/password
    except (FileNotFoundError, TypeError):
        print('Please provide Earthdata Login credentials for access.')
        username, password = input('Username: '), getpass('Password: ')
    
    manager = request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, url, username, password)
    auth = request.HTTPBasicAuthHandler(manager)
    jar = CookieJar()
    processor = request.HTTPCookieProcessor(jar)
    opener = request.build_opener(auth, processor)
    request.install_opener(opener)

################################################################################################
##                                         Constants                                          ##
################################################################################################
# Seawater density (kg/m^3)
rhoconst = 1029     ## needed to convert surface mass fluxes to volume fluxes
## needed to convert surface mass fluxes to volume fluxes

# Heat capacity (J/kg/K)
c_p = 3994

# Constants for surface heat penetration (from Table 2 of Paulson and Simpson, 1977)
R = 0.62
zeta1 = 0.6
zeta2 = 20.0


download_root_dir = Path('/Users/joegradone/SynologyDrive/Drive/Rutgers/Research/data/ECCO/')


################################################################################################

for year_start in np.arange(1993,2017):
    
    # define year bounds [year_start,year_end)...year_end is excluded
    #year_start = 1993
    year_end = year_start+1
    
    # directories where snapshots are located
    etan_snaps_dir = join(download_root_dir,'ECCO_L4_SSH_LLC0090GRID_SNAPSHOT_V4R4')
    salt_snaps_dir = join(download_root_dir,'ECCO_L4_TEMP_SALINITY_LLC090GRID_SNAPSHOT_V4R4')
    
    # define function to get list of files in year range
    def files_in_year_range(file_dir,year_start,year_end):
        "Creates text list of files in the year range [year_start,year_end)"
        files_in_range = []
        for file in os.listdir(file_dir):
        # use regex search to find year associated with file
            if len(file)>10:
                year_match = re.search('[0-9]{4}(?=-[0-9]{2})',file)
                curr_yr = int(year_match.group(0))
                if (curr_yr >= year_start) and (curr_yr < year_end):
                    files_in_range.append(join(file_dir,file))
        return files_in_range
    
    # get list of files in year range (including year_end)
    etan_snaps_in_range = files_in_year_range(etan_snaps_dir,year_start,year_end+1)
    salt_snaps_in_range = files_in_year_range(salt_snaps_dir,year_start,year_end+1)
    
    # open files as two xarray datasets, then merge to create one dataset with the variables we need
    ds_etan_snaps = xr.open_mfdataset(etan_snaps_in_range,\
                                           data_vars='minimal',coords='minimal',compat='override')
    ds_salt_snaps = xr.open_mfdataset(salt_snaps_in_range,\
                                           data_vars='minimal',coords='minimal',compat='override')
    ecco_monthly_snaps = xr.merge([ds_etan_snaps['ETAN'],ds_salt_snaps[['SALT','THETA']]])
    
    # Exclude snapshots after Jan 1 of year_end
    ecco_monthly_snaps = ecco_monthly_snaps.isel(time=np.arange(0, 13))
    
    
    
    ################################################################################################
    fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_FRESH_FLUX_LLC0090GRID_MONTHLY_V4R4/*',str(year_start),'*.nc']))
    fresh_flux = xr.open_mfdataset(fnames,data_vars='minimal',coords='minimal',compat='override')
    
    fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_OCEAN_3D_SALINITY_FLUX_LLC0090GRID_MONTHLY_V4R4/*',str(year_start),'*.nc']))
    salt_flux = xr.open_mfdataset(fnames,data_vars='minimal',coords='minimal',compat='override')
    
    fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_HEAT_FLUX_LLC0090GRID_MONTHLY_V4R4/*',str(year_start),'*.nc']))
    heat_flux = xr.open_mfdataset(fnames,data_vars='minimal',coords='minimal',compat='override')
    
    fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_OCEAN_3D_TEMPERATURE_FLUX_LLC0090GRID_MONTHLY_V4R4/*',str(year_start),'*.nc']))
    temp_flux = xr.open_mfdataset(fnames,data_vars='minimal',coords='minimal',compat='override')
    
    fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_OCEAN_3D_VOLUME_FLUX_LLC0090GRID_MONTHLY_V4R4/*',str(year_start),'*.nc']))
    vol_flux = xr.open_mfdataset(fnames,data_vars='minimal',coords='minimal',compat='override')
    
    fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_OCEAN_VEL_LLC0090GRID_MONTHLY_V4R4/*',str(year_start),'*.nc']))
    vel = xr.open_mfdataset(fnames,data_vars='minimal',coords='minimal',compat='override')
    
    fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_OCEAN_BOLUS_STREAMFUNCTION_LLC0090GRID_MONTHLY_V4R4/*',str(year_start),'*.nc']))
    bolus = xr.open_mfdataset(fnames,data_vars='minimal',coords='minimal',compat='override')
    
    fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_MIXED_LAYER_DEPTH_LLC0090GRID_MONTHLY_V4R4/*',str(year_start),'*.nc']))
    mld_ds = xr.open_mfdataset(fnames,data_vars='minimal',coords='minimal',compat='override')
    
    fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_SSH_LLC0090GRID_MONTHLY_V4R4/*',str(year_start),'*.nc']))
    etan_monthly = xr.open_mfdataset(fnames,data_vars='minimal',coords='minimal',compat='override')
    
    fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_TEMP_SALINITY_LLC0090GRID_MONTHLY_V4R4/*',str(year_start),'*.nc']))
    ts_monthly = xr.open_mfdataset(fnames,data_vars='minimal',coords='minimal',compat='override')
    
    
    ###############################################################################################
    ecco_monthly_mean = xr.merge([fresh_flux[['SFLUX','oceFWflx']],
                      salt_flux[['oceSPtnd','ADVx_SLT','ADVy_SLT','ADVr_SLT','DFxE_SLT','DFyE_SLT','DFrE_SLT','DFrI_SLT']],
                      heat_flux[['oceQsw','TFLUX']],
                      temp_flux[['ADVx_TH','ADVy_TH','ADVr_TH','DFxE_TH','DFyE_TH','DFrE_TH','DFrI_TH']],
                      vol_flux[['UVELMASS','VVELMASS','WVELMASS']],
                      vel[['UVEL','VVEL','WVEL']],
                      bolus[['GM_PsiX','GM_PsiY']],
                      etan_monthly['ETAN'],
                      ts_monthly[['SALT','THETA']]])
    
    ecco_monthly_mean = ecco_monthly_mean.reset_coords(drop=True)
    
    ds = xr.merge([ecco_monthly_mean,
                   ecco_monthly_snaps.rename({'time':'time_snp','ETAN':'ETAN_snp', 'SALT':'SALT_snp', 'THETA':'THETA_snp'})])\
                   .chunk({'time':1,'time_snp':1})
    
    
    ################################################################################################
    ## Now actually do the budgets
    ################################################################################################
    ## Load grid
    ecco_grid = xr.open_dataset('/Users/joegradone/SynologyDrive/Drive/Rutgers/Research/data/ECCO/ECCO_L4_GEOMETRY_LLC0090GRID_V4R4/GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc')
    
    # Volume (m^3)
    vol = (ecco_grid.rA*ecco_grid.drF*ecco_grid.hFacC)
    
    ## Predefine coordinates for global regridding of the ECCO output (used in resample_to_latlon)
    new_grid_delta_lat = 1
    new_grid_delta_lon = 1
    
    new_grid_min_lat = -90
    new_grid_max_lat = 90
    
    new_grid_min_lon = -180
    new_grid_max_lon = 180
    
    
    # Change time axis of the snapshot variables
    ds.time_snp.attrs['c_grid_axis_shift'] = 0.5
    grid = ecco.get_llc_grid(ds)
    
    #grid = xgcm.Grid(ds)
    # time delta
    delta_t = grid.diff(ds.time_snp, 'T', boundary='fill', fill_value=np.nan)
    delta_t = delta_t.astype('f4') / 1e9
    
    
    ################################################################################################
    ## Calculate the total salt and temperature tendency
    ################################################################################################
    
    # Calculate the s*S term
    sSALT = ds.SALT_snp*(1+ds.ETAN_snp/ecco_grid.Depth)
    # Calculate the s*theta term
    sTHETA = ds.THETA_snp*(1+ds.ETAN_snp/ecco_grid.Depth)
    
    # Total tendency (psu/s)
    sSALT_diff = sSALT.diff('time_snp')
    sSALT_diff = sSALT_diff.rename({'time_snp':'time'})
    del sSALT_diff.time.attrs['c_grid_axis_shift']    # remove attribute from DataArray
    sSALT_diff = sSALT_diff.assign_coords(time=ds.time)    # correct time coordinate values
    G_total_Slt = sSALT_diff/delta_t
    
    # Total tendency (deg C/s)
    sTHETA_diff = sTHETA.diff('time_snp')
    sTHETA_diff = sTHETA_diff.rename({'time_snp':'time'})
    del sTHETA_diff.time.attrs['c_grid_axis_shift']    # remove attribute from DataArray
    sTHETA_diff = sTHETA_diff.assign_coords(time=ds.time)    # correct time coordinate values
    G_total_TH = sTHETA_diff/delta_t
    
    
    ################################################################################################
    ## Calculate tendency due to advective convergence
    ################################################################################################
    ### Horizontal convergence of advective heat/salt flux
    # Set fluxes on land to zero (instead of NaN)
    # ds['ADVx_SLT'] = ds.ADVx_SLT.where(ecco_grid.hFacW.expand_dims(dim={"time": len(ds.time)}).values > 0,0)
    # ds['ADVy_SLT'] = ds.ADVy_SLT.where(ecco_grid.hFacS.expand_dims(dim={"time": len(ds.time)}).values > 0,0)
    # ds['ADVx_TH'] = ds.ADVx_TH.where(ecco_grid.hFacW.expand_dims(dim={"time": len(ds.time)}).values > 0,0)
    # ds['ADVy_TH'] = ds.ADVy_TH.where(ecco_grid.hFacS.expand_dims(dim={"time": len(ds.time)}).values > 0,0)
    ds['ADVx_SLT'] = ds.ADVx_SLT.where(ecco_grid.hFacW.values > 0,0)
    ds['ADVy_SLT'] = ds.ADVy_SLT.where(ecco_grid.hFacS.values > 0,0)
    ds['ADVx_TH'] = ds.ADVx_TH.where(ecco_grid.hFacW.values > 0,0)
    ds['ADVy_TH'] = ds.ADVy_TH.where(ecco_grid.hFacS.values > 0,0)
    
    # compute horizontal components of divergence
    ADVxy_diff_SLT = grid.diff_2d_vector({'X' : ds.ADVx_SLT, 'Y' : ds.ADVy_SLT}, boundary = 'fill')
    ADVxy_diff_TH = grid.diff_2d_vector({'X' : ds.ADVx_TH, 'Y' : ds.ADVy_TH}, boundary = 'fill')
    
    # Convergence of horizontal advection (psu m^3/s)
    adv_hConvS = (-(ADVxy_diff_SLT['X'] + ADVxy_diff_SLT['Y']))
    adv_hConvS = adv_hConvS/vol
    adv_hConvSx = -(ADVxy_diff_SLT['X'] )/vol
    adv_hConvSy = -(ADVxy_diff_SLT['Y'] )/vol
    
    # Convergence of horizontal advection (degC m^3/s)
    adv_hConvH = (-(ADVxy_diff_TH['X'] + ADVxy_diff_TH['Y']))
    adv_hConvH = adv_hConvH/vol
    adv_hConvHx = -(ADVxy_diff_TH['X'])/vol
    adv_hConvHy = -(ADVxy_diff_TH['Y'])/vol
    
    ### Vertical convergence of advective heat/salt flux
    
    # Set fluxes on land to zero (instead of NaN)
    # ds['ADVr_SLT'] = ds.ADVr_SLT.where(ecco_grid.hFacC.expand_dims(dim={"time": len(ds.time)}).values > 0,0)
    # ds['ADVr_TH'] = ds.ADVr_TH.where(ecco_grid.hFacC.expand_dims(dim={"time": len(ds.time)}).values > 0,0)
    ds['ADVr_SLT'] = ds.ADVr_SLT.where(ecco_grid.hFacC.values > 0,0)
    ds['ADVr_TH'] = ds.ADVr_TH.where(ecco_grid.hFacC.values > 0,0)
    
    # Need to make sure that sequence of dimensions are consistent
    ADVr_TH = ds.ADVr_TH.transpose('time','tile','k_l','j','i')
    
    # Convergence of vertical advection (psu/s)
    adv_vConvS = grid.diff(ds.ADVr_SLT, 'Z', boundary='fill')
    adv_vConvS = adv_vConvS/vol
    
    # Convergence of vertical advection (degC m^3/s)
    adv_vConvH = grid.diff(ADVr_TH, 'Z', boundary='fill')
    adv_vConvH = adv_vConvH/vol
    
    ### Total convergence of advective heat/salt flux
    
    # Sum horizontal and vertical convergences and divide by volume (degC/s)
    G_advection_TH = (adv_hConvH + adv_vConvH)
    
    # Sum horizontal and vertical convergences and divide by volume (psu/s)
    G_advection_Slt = (adv_hConvS + adv_vConvS)
    
    
    ################################################################################################
    ## Calculate tendency due to diffusive convergence
    ################################################################################################
    ### Horizontal convergence of diffusive heat/salt flux
    
    # Set fluxes on land to zero (instead of NaN)
    # ds['DFxE_SLT'] = ds.DFxE_SLT.where(ecco_grid.hFacW.expand_dims(dim={"time": len(ds.time)}).values > 0,0)
    # ds['DFyE_SLT'] = ds.DFyE_SLT.where(ecco_grid.hFacS.expand_dims(dim={"time": len(ds.time)}).values > 0,0)
    # ds['DFxE_TH'] = ds.DFxE_TH.where(ecco_grid.hFacW.expand_dims(dim={"time": len(ds.time)}).values > 0,0)
    # ds['DFyE_TH'] = ds.DFyE_TH.where(ecco_grid.hFacS.expand_dims(dim={"time": len(ds.time)}).values > 0,0)
    ds['DFxE_SLT'] = ds.DFxE_SLT.where(ecco_grid.hFacW.values > 0,0)
    ds['DFyE_SLT'] = ds.DFyE_SLT.where(ecco_grid.hFacS.values > 0,0)
    ds['DFxE_TH'] = ds.DFxE_TH.where(ecco_grid.hFacW.values > 0,0)
    ds['DFyE_TH'] = ds.DFyE_TH.where(ecco_grid.hFacS.values > 0,0)
    
    DFxyE_diff_SLT = grid.diff_2d_vector({'X' : ds.DFxE_SLT, 'Y' : ds.DFyE_SLT}, boundary = 'fill')
    DFxyE_diff_TH = grid.diff_2d_vector({'X' : ds.DFxE_TH, 'Y' : ds.DFyE_TH}, boundary = 'fill')
    
    # Convergence of horizontal diffusion (psu m^3/s)
    dif_hConvS = (-(DFxyE_diff_SLT['X'] + DFxyE_diff_SLT['Y']))
    dif_hConvS = dif_hConvS/vol
    dif_hConvSx = -(DFxyE_diff_SLT['X'])/vol
    dif_hConvSy = -(DFxyE_diff_SLT['Y'])/vol
    
    # Convergence of horizontal diffusion (degC m^3/s)
    dif_hConvH = (-(DFxyE_diff_TH['X'] + DFxyE_diff_TH['Y']))
    dif_hConvH = dif_hConvH/vol
    dif_hConvHx = -(DFxyE_diff_TH['X'])/vol
    dif_hConvHy = -(DFxyE_diff_TH['Y'])/vol
    
    
    ### Vertical convergence of diffusive heat/salt flux
    
    # Set fluxes on land to zero (instead of NaN)
    # ds['DFrE_TH'] = ds.DFrE_TH.where(ecco_grid.hFacC.expand_dims(dim={"time": len(ds.time)}).values > 0,0)
    # ds['DFrI_TH'] = ds.DFrI_TH.where(ecco_grid.hFacC.expand_dims(dim={"time": len(ds.time)}).values > 0,0)
    # ds['DFrE_SLT'] = ds.DFrE_SLT.where(ecco_grid.hFacC.expand_dims(dim={"time": len(ds.time)}).values > 0,0)
    # ds['DFrI_SLT'] = ds.DFrI_SLT.where(ecco_grid.hFacC.expand_dims(dim={"time": len(ds.time)}).values > 0,0)
    ds['DFrE_TH'] = ds.DFrE_TH.where(ecco_grid.hFacC.values > 0,0)
    ds['DFrI_TH'] = ds.DFrI_TH.where(ecco_grid.hFacC.values > 0,0)
    ds['DFrE_SLT'] = ds.DFrE_SLT.where(ecco_grid.hFacC.values > 0,0)
    ds['DFrI_SLT'] = ds.DFrI_SLT.where(ecco_grid.hFacC.values > 0,0)
    
    # Load monthly averages of vertical diffusive fluxes
    DFrE_TH = ds.DFrE_TH.transpose('time','tile','k_l','j','i')
    DFrI_TH = ds.DFrI_TH.transpose('time','tile','k_l','j','i')
    
    # Convergence of vertical diffusion (degC m^3/s)
    dif_vConvH = grid.diff(DFrE_TH, 'Z', boundary='fill') + grid.diff(DFrI_TH, 'Z', boundary='fill')
    dif_vConvH = dif_vConvH/vol
    
    # Convergence of vertical diffusion (psu m^3/s)
    dif_vConvS = grid.diff(ds.DFrE_SLT, 'Z', boundary='fill') + grid.diff(ds.DFrI_SLT, 'Z', boundary='fill')
    dif_vConvS = dif_vConvS/vol
    
    ### Total convergence of diffusive heat/salt flux
    
    # Sum horizontal and vertical convergences and divide by volume (degC/s)
    G_diffusion_TH = (dif_hConvH + dif_vConvH)
    # Sum horizontal and vertical convergences and divide by volume (psu/s)
    G_diffusion_Slt = (dif_hConvS + dif_vConvS)
    
    
    ################################################################################################
    ## Calculate the tendency due to forcing
    ################################################################################################
    
    ### Surface heat first
    Z = ecco_grid.Z.compute()
    RF = np.concatenate([ecco_grid.Zp1.values[:-1],[np.nan]])
    q1 = R*np.exp(1.0/zeta1*RF[:-1]) + (1.0-R)*np.exp(1.0/zeta2*RF[:-1])
    q2 = R*np.exp(1.0/zeta1*RF[1:]) + (1.0-R)*np.exp(1.0/zeta2*RF[1:])
    # Correction for the 200m cutoff
    zCut = np.where(Z < -200)[0][0]
    q1[zCut:] = 0
    q2[zCut-1:] = 0
    # Create xarray data arrays
    q1 = xr.DataArray(q1,coords=[Z.k],dims=['k'])
    q2 = xr.DataArray(q2,coords=[Z.k],dims=['k'])
    
    ### Compute vertically penetrating flux
    
    ## Land masks
    # Make copy of hFacC
    mskC = ecco_grid.hFacC.copy(deep=True).compute()
    
    # Change all fractions (ocean) to 1. land = 0
    mskC.values[mskC.values>0] = 1
    
    # Shortwave flux below the surface (W/m^2)
    forcH_subsurf = ((q1*(mskC==1)-q2*(mskC.shift(k=-1)==1))*ds.oceQsw).transpose('time','tile','k','j','i')
    
    # Surface heat flux (W/m^2)
    forcH_surf = ((ds.TFLUX - (1-(q1[0]-q2[0]))*ds.oceQsw)\
                  *mskC[0]).transpose('time','tile','j','i').assign_coords(k=0).expand_dims('k')
        
    # Full-depth sea surface forcing (W/m^2)
    forcH = xr.concat([forcH_surf,forcH_subsurf[:,:,1:]], dim='k').transpose('time','tile','k','j','i')
    
    ### Total heat forcing
    
    # Add geothermal heat flux to forcing field and convert from W/m^2 to degC/s
    G_forcing_TH = ((forcH)/(rhoconst*c_p))/(ecco_grid.hFacC*ecco_grid.drF)
    
    ### Now salt forcing
    # Load SFLUX and add vertical coordinate
    SFLUX = ds.SFLUX.assign_coords(k=0).expand_dims(dim='k',axis=1)
    
    # Calculate forcing term by adding SFLUX and oceSPtnd (g/m^2/s)
    forcS = xr.concat([SFLUX+ds.oceSPtnd,ds.oceSPtnd.isel(k=slice(1,None))], dim='k')
    # Forcing (psu/s)
    G_forcing_Slt = forcS/rhoconst/(ecco_grid.hFacC*ecco_grid.drF)
    
    salt = ds.SALT
    temp = ds.THETA
    uvel = ds.UVEL
    vvel = ds.VVEL
    wvel = ds.WVEL
    
    ################################################################################################
    ##                                         Salinity Budget                                    ##
    ################################################################################################
    # Scale factor
    rstarfac = ((ecco_grid.Depth + ds.ETAN)/ecco_grid.Depth)
    
    
    # Total tendency (psu/s)
    SALT_diff = ds.SALT_snp.diff('time_snp')
    SALT_diff = SALT_diff.rename({'time_snp':'time'})
    del SALT_diff.time.attrs['c_grid_axis_shift']    # remove attribute from DataArray
    SALT_diff = SALT_diff.assign_coords(time=ds.time)    # correct time coordinate values
    G_total_Sln = SALT_diff/delta_t
    
    
    ################################################################################################
    ## Calculate tendency due to advective convergence
    ################################################################################################
    ### H# Horizontal volume transports (m^3/s)
    u_transport = ds.UVELMASS * ecco_grid.dyG * ecco_grid.drF
    v_transport = ds.VVELMASS * ecco_grid.dxG * ecco_grid.drF
    
    # Set fluxes on land to zero (instead of NaN)
    u_transport = u_transport.where(ecco_grid.hFacW.values > 0,0)
    v_transport = v_transport.where(ecco_grid.hFacS.values > 0,0)
    
    uv_diff = grid.diff_2d_vector({'X' : u_transport, 'Y' : v_transport}, boundary = 'fill')
    
    # Convergence of the horizontal flow (m^3/s)
    hConvV = -(uv_diff['X'] + uv_diff['Y'])
    
    # Horizontal convergence of salinity (m^3/s)
    adv_hConvSln = ((-ds.SALT*hConvV + adv_hConvS)/rstarfac)/vol
    
    # Vertical volume transport (m^3/s)
    w_transport = ds.WVELMASS.where(ds.k_l>0,0.) * ecco_grid.rA
    
    # Set land values of flux to zero (instead of NaN)
    w_transport = w_transport.where(ecco_grid.hFacC.values > 0,0)
    
    # Convergence of the vertical flow (m^3/s)
    vConvV = grid.diff(w_transport, 'Z', boundary='fill')
    
    # Vertical convergence of salinity (psu m^3/s)
    adv_vConvSln = ((-ds.SALT*vConvV + adv_vConvS)/rstarfac)/vol
    
    # Total convergence of advective salinity flux (psu/s)
    G_advection_Sln = (adv_hConvSln + adv_vConvSln)
    
    ################################################################################################
    ## Calculate tendency due to diffusive convergence
    ################################################################################################
    # Horizontal convergence
    dif_hConvSln = (dif_hConvS/rstarfac)/vol
    
    # Vertical convergence
    dif_vConvSln = (dif_vConvS/rstarfac)/vol
    
    # Sum horizontal and vertical convergences and divide by volume (psu/s)
    G_diffusion_Sln = (dif_hConvSln + dif_vConvSln)
    
    ################################################################################################
    ## Calculate tendency due to forcing
    ################################################################################################
    # Load monthly averaged freshwater flux and add vertical coordinate
    oceFWflx = ds.oceFWflx.assign_coords(k=0).expand_dims('k')
    
    # Sea surface forcing on volume (1/s)
    forcV = xr.concat([(oceFWflx/rhoconst)/(ecco_grid.hFacC*ecco_grid.drF), 
                        xr.zeros_like((oceFWflx[0]/rhoconst)/(ecco_grid.hFacC*ecco_grid.drF).isel(k=slice(1,None)))], 
                      dim='k')
    
    # Sea surface forcing for salinity (psu/s)
    G_forcing_Sln = (-ds.SALT*forcV + G_forcing_Slt)/rstarfac
    
    
    ################################################################################################
    ##                                       Freshwater Budget                                    ##
    ################################################################################################
    # Reference salinity
    Sref = 35.0
    
    f = (Sref - ds.SALT_snp)/Sref
    
    # Total freshwater tendency (m^3/s)
    f_diff = f.diff('time_snp')
    f_diff = f_diff.rename({'time_snp':'time'})
    del f_diff.time.attrs['c_grid_axis_shift']    # remove attribute from DataArray
    f_diff = f_diff.assign_coords(time=ds.time)    # correct time coordinate values
    G_total_Fw = f_diff*vol/delta_t
    
    # Set values on land to zero (instead of NaN)
    ds['GM_PsiX'] = ds.GM_PsiX.where(ecco_grid.hFacW.values > 0,0)
    ds['GM_PsiY'] = ds.GM_PsiY.where(ecco_grid.hFacS.values > 0,0)
    
    UVELSTAR = grid.diff(ds.GM_PsiX, 'Z', boundary='fill')/ecco_grid.drF
    VVELSTAR = grid.diff(ds.GM_PsiY, 'Z', boundary='fill')/ecco_grid.drF
    
    GM_PsiXY_diff = grid.diff_2d_vector({'X' : ds.GM_PsiX*ecco_grid.dyG, 
                                          'Y' : ds.GM_PsiY*ecco_grid.dxG}, boundary = 'fill')
    WVELSTAR = (GM_PsiXY_diff['X'] + GM_PsiXY_diff['Y'])/ecco_grid.rA
    
    
    ################################################################################################
    ## Advective freshwater flux
    ################################################################################################
    SALT_at_u = grid.interp(ds.SALT, 'X', boundary='extend')
    SALT_at_v = grid.interp(ds.SALT, 'Y', boundary='extend')
    SALT_at_w = grid.interp(ds.SALT, 'Z', boundary='extend')
    # Freshwater advective (Eulerian+Bolus) fluxes (m^3/s)
    ADVx_FW = (ds.UVELMASS+UVELSTAR)*ecco_grid.dyG*ecco_grid.drF*(Sref-SALT_at_u)/Sref
    ADVy_FW = (ds.VVELMASS+VVELSTAR)*ecco_grid.dxG*ecco_grid.drF*(Sref-SALT_at_v)/Sref
    ADVr_FW = (ds.WVELMASS.where(ds.k_l>0).fillna(0.)+WVELSTAR)*ecco_grid.rA*(Sref-SALT_at_w)/Sref
    # set fluxes on land to zero (instead of NaN)
    ADVx_FW = ADVx_FW.where(ecco_grid.hFacW.values > 0,0)
    ADVy_FW = ADVy_FW.where(ecco_grid.hFacS.values > 0,0)
    ADVr_FW = ADVr_FW.where(ecco_grid.hFacC.values > 0,0)
    
    # We calculate the diffusive freshwater flux as the residuals of the remaining budget terms as this term is not output by ECCO
    ADVxy_diff = grid.diff_2d_vector({'X' : ADVx_FW, 'Y' : ADVy_FW}, boundary = 'fill')
    
    # Convergence of horizontal advection (m^3/s)
    adv_hConvFw = (-(ADVxy_diff['X'] + ADVxy_diff['Y']))
    # Convergence of vertical advection (m^3/s)
    adv_vConvFw = grid.diff(ADVr_FW, 'Z', boundary='fill')
    # Sum horizontal and vertical convergences (m^3/s)
    G_advection_Fw = (adv_hConvFw + adv_vConvFw)/rstarfac
    
    ################################################################################################
    ## Freshwater forcing
    ################################################################################################
    # Freshwater forcing (m^3/s)
    forcFw = ds.oceFWflx/rhoconst*ecco_grid.rA
    
    # Expand to fully 3d (using G_advection_Fw as template)
    forcing_Fw = xr.concat([forcFw.reset_coords(drop=True).assign_coords(k=0).expand_dims('k'),
                              xr.zeros_like(G_advection_Fw).isel(k=slice(1,None))],
                              dim='k').where(ecco_grid.hFacC==1)
    # Sum FW and Salinity forcing, changing G_forcing_Slt from [m psu/s] to [m^3/s]
    G_forcing_Fw = (forcing_Fw-G_forcing_Slt*ecco_grid.rA/rhoconst/Sref)/rstarfac
    
    # Convergence of freshwater diffusion (m^3/s)
    G_diffusion_Fw = G_total_Fw - G_forcing_Fw - G_advection_Fw
    
    
    ################################################################################################
    ## Prep to save
    ################################################################################################
    
    mld = mld_ds.MXLDEPTH
        
    varnames = ['G_total_TH','adv_hConvH','adv_vConvH','dif_hConvH','dif_vConvH','G_forcing_TH',
                'G_total_Slt','adv_hConvS','adv_vConvS','dif_hConvS','dif_vConvS','G_forcing_Slt',
                'G_total_Sln','adv_hConvSln','adv_vConvSln','dif_hConvSln','dif_vConvSln','G_forcing_Sln',
                'G_total_Fw', 'G_advection_Fw', 'G_diffusion_Fw', 'G_forcing_Fw',
                'salt','temp','uvel','vvel','wvel','mld']
    
    ds2 = xr.Dataset(data_vars={})
    
    for varname in varnames:
        ds2[varname] = globals()[varname].chunk(chunks='auto')
    
    # Add surface forcing (degC/s)
    ds2['Qnet'] = ((forcH /(rhoconst*c_p))\
                  /(ecco_grid.hFacC*ecco_grid.drF)).chunk(chunks='auto')
    
    # Add shortwave penetrative flux (degC/s)
    #Since we only are interested in the subsurface heat flxux we need to zero out the top cell
    SWpen = ((forcH_subsurf /(rhoconst*c_p))/(ecco_grid.hFacC*ecco_grid.drF)).where(forcH_subsurf.k>0).fillna(0.)
    ds2['SWpen'] = SWpen.where(ecco_grid.hFacC>0).chunk(chunks='auto')
    
    ds2.time.encoding = {}
    ds2 = ds2.reset_coords(drop=True)
    
    
    ### Before this step, I pause and delete all other global variables in memory. It seems to help
    ### prevent the kernel from crashing
    print('Saving year: ',year_start)
    
    with ProgressBar():
         ds2.to_netcdf('/Users/joegradone/SynologyDrive/Drive/Rutgers/Research/data/ECCO/Evaluated_Budgets/eccov4r4_budget_heat_salt_1993_2016/'+'eccov4r4_budg_heat_salt_'+str(year_start)+'.nc')
    
