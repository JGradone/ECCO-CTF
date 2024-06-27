# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import datetime
sys.path.append('/Users/joegradone/SynologyDrive/Drive/Rutgers/Research/code/GitHub/ECCOv4-py')
import ecco_v4_py as ecco
#from ecco_download import *
import numpy as np
import pandas as pd
from os.path import join,expanduser
import xarray as xr
import matplotlib.pyplot as plt
from getpass import getpass
from http.cookiejar import CookieJar
from pathlib import Path
from netrc import netrc
from cartopy.mpl.geoaxes import GeoAxes
import cartopy
import geopandas
from scipy.stats import linregress
from mpl_toolkits.axes_grid1 import AxesGrid
# library to download files
from urllib import request
from scipy import signal
import glob
import xgcm
import scipy.signal as sps
import scipy.linalg as spl
from xgcm import Grid
import cmocean.cm as cmo
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
## For detrending the xarray dataset, from: https://xrft.readthedocs.io/en/latest/_modules/xrft/detrend.html

def detrend(da, dim, detrend_type="constant"):
    """
    Detrend a DataArray

    Parameters
    ----------
    da : xarray.DataArray
        The data to detrend
    dim : str or list
        Dimensions along which to apply detrend.
        Can be either one dimension or a list with two dimensions.
        Higher-dimensional detrending is not supported.
        If dask data are passed, the data must be chunked along dim.
    detrend_type : {'constant', 'linear'}
        If ``constant``, a constant offset will be removed from each dim.
        If ``linear``, a linear least-squares fit will be estimated and removed
        from the data.

    Returns
    -------
    da : xarray.DataArray
        The detrended data.

    Notes
    -----
    This function will act lazily in the presence of dask arrays on the
    input.
    """

    if dim is None:
        dim = list(da.dims)
    else:
        if isinstance(dim, str):
            dim = [dim]

    if detrend_type not in ["constant", "linear", None]:
        raise NotImplementedError(
            "%s is not a valid detrending option. Valid "
            "options are: 'constant','linear', or None." % detrend_type
        )

    if detrend_type is None:
        return da
    elif detrend_type == "constant":
        return da - da.mean(dim=dim)
    elif detrend_type == "linear":
        data = da.data
        axis_num = [da.get_axis_num(d) for d in dim]
        chunks = getattr(data, "chunks", None)
        if chunks:
            axis_chunks = [data.chunks[a] for a in axis_num]
            if not all([len(ac) == 1 for ac in axis_chunks]):
                raise ValueError("Contiguous chunks required for detrending.")
        if len(dim) == 1:
            dt = xr.apply_ufunc(
                sps.detrend,
                da,
                axis_num[0],
                output_dtypes=[da.dtype],
                dask="parallelized",
            )
        elif len(dim) == 2:
            dt = xr.apply_ufunc(
                _detrend_2d_ufunc,
                da,
                input_core_dims=[dim],
                output_core_dims=[dim],
                output_dtypes=[da.dtype],
                vectorize=True,
                dask="parallelized",
            )
        else:  # pragma: no cover
            raise NotImplementedError(
                "Only 1D and 2D detrending are implemented so far."
            )

    return dt


################################################################################################
## For the lowpass filter
from scipy.signal import butter, sosfilt, sosfreqz

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y




################################################################################################
fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_FRESH_FLUX_LLC0090GRID_DAILY_V4R4/*.nc']))
fresh_flux = xr.open_mfdataset(fnames, parallel=True, data_vars='minimal', coords='minimal', compat='override')

fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_OCEAN_3D_SALINITY_FLUX_LLC0090GRID_DAILY_V4R4/*.nc']))
salt_flux = xr.open_mfdataset(fnames, parallel=True, data_vars='minimal', coords='minimal', compat='override')

fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_HEAT_FLUX_LLC0090GRID_DAILY_V4R4/*.nc']))
heat_flux = xr.open_mfdataset(fnames, parallel=True, data_vars='minimal', coords='minimal', compat='override')

fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_OCEAN_3D_TEMPERATURE_FLUX_LLC0090GRID_DAILY_V4R4/*.nc']))
temp_flux = xr.open_mfdataset(fnames, parallel=True, data_vars='minimal', coords='minimal', compat='override')

fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_OCEAN_3D_VOLUME_FLUX_LLC0090GRID_DAILY_V4R4/*.nc']))
vol_flux = xr.open_mfdataset(fnames, parallel=True, data_vars='minimal', coords='minimal', compat='override')

fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_OCEAN_BOLUS_STREAMFUNCTION_LLC0090GRID_DAILY_V4R4/*.nc']))
bolus = xr.open_mfdataset(fnames, parallel=True, data_vars='minimal', coords='minimal', compat='override')

fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_SSH_LLC0090GRID_DAILY_V4R4/*.nc']))
etan = xr.open_mfdataset(fnames, parallel=True, data_vars='minimal', coords='minimal', compat='override')

fnames = glob.glob(''.join([str(download_root_dir),'/ECCO_L4_TEMP_SALINITY_LLC0090GRID_DAILY_V4R4/*.nc']))
ts = xr.open_mfdataset(fnames, parallel=True, data_vars='minimal', coords='minimal', compat='override')
################################################################################################



################################################################################################
ds = xr.merge([fresh_flux[['SFLUX','oceFWflx']],
                 salt_flux[['oceSPtnd','ADVx_SLT','ADVy_SLT','ADVr_SLT','DFxE_SLT','DFyE_SLT','DFrE_SLT','DFrI_SLT']],
                 heat_flux[['oceQsw','TFLUX']],
                 temp_flux[['ADVx_TH','ADVy_TH','ADVr_TH','DFxE_TH','DFyE_TH','DFrE_TH','DFrI_TH']],
                 vol_flux[['UVELMASS','VVELMASS','WVELMASS']],
                 bolus[['GM_PsiX','GM_PsiY']],
                 etan['ETAN'],
                 ts[['SALT','THETA']]])

ds = ds.rename({'time':'time_snp','ETAN':'ETAN_snp', 'SALT':'SALT_snp','THETA':'THETA_snp'})
# Drop superfluous coordinates (We already have them in ecco_grid)
ds = ds.reset_coords(drop=True)
## Remove leap day
ds = ds.sel(time_snp=~((ds.time_snp.dt.month == 2) & (ds.time_snp.dt.day == 29)))
## Remove all=time mean
ds = ds-ds.mean(dim='time_snp')
## Detrend
ds = detrend(ds, dim='time_snp', detrend_type="constant")

## Steps to remove daily climatological seasonal cycle
doymean = ds.groupby(ds.time_snp.dt.dayofyear).mean()

## Pull out timestamps because we're doing to write over this for the purposes of the lowpass filter
time_snp = ds['time_snp'].values
## Write over with just day counts
ds['time_snp'] = np.arange(0,len(ds.time_snp),1)

## Remove leap day
doymean = doymean.where(doymean.dayofyear!=60,drop=True)
tot_doymean = xr.concat([doymean,doymean,doymean,doymean,doymean,doymean,doymean,doymean,doymean,doymean,doymean,doymean,doymean], dim='dayofyear')
## Make sure dimensions have same name for the subtraction
tot_doymean = tot_doymean.rename({'dayofyear': 'time_snp'})
tot_doymean['time_snp'] = np.arange(0,len(tot_doymean.time_snp),1)
## Do the damn thing
ds = ds-tot_doymean

## Bandpass filter
fs = 1
lowcut = 1/150 ## 150 days
highcut = 1/30 ## 30 days
ds = xr.apply_ufunc(butter_bandpass_filter, ds.chunk(dict(time_snp=-1)), input_core_dims=[["time_snp"]],output_core_dims=[["time_snp"]], kwargs={'fs': fs,'lowcut':lowcut,'highcut':highcut}, dask="parallelized", vectorize = True)


## Load grid
ecco_grid = xr.open_dataset('/Users/joegradone/SynologyDrive/Drive/Rutgers/Research/data/ECCO/ECCO_L4_GEOMETRY_LLC0090GRID_V4R4/GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc')
ecco_grid = ecco_grid.isel(tile=[10,11])



################################################################################################
## Now actually do the budgets
################################################################################################

# Volume (m^3)
vol = (ecco_grid.rA*ecco_grid.drF*ecco_grid.hFacC)

## Predefine coordinates for global regridding of the ECCO output (used in resample_to_latlon)
new_grid_delta_lat = 1
new_grid_delta_lon = 1

new_grid_min_lat = -90
new_grid_max_lat = 90

new_grid_min_lon = -180
new_grid_max_lon = 180

## Create the xgcm ‘grid’ object
# Change time axis of the snapshot variables
ds.time_snp.attrs['c_grid_axis_shift'] = 0.5
grid = xgcm.Grid(ecco_grid)
# time delta
delta_t = (time_snp[1] - time_snp[0])/ np.timedelta64(1, 's')

################################################################################################
## Calculate the total salt and temperature tendency
################################################################################################

# Calculate the s*S term
sSALT = ds.SALT_snp*(1+ds.ETAN_snp/ecco_grid.Depth)
# Calculate the s*theta term
sTHETA = ds.THETA_snp*(1+ds.ETAN_snp/ecco_grid.Depth)

# Total tendency (psu/s)
sSALT_diff = sSALT.diff('time_snp')
#sSALT_diff = sSALT_diff.rename({'time':'time_snp'})
#del sSALT_diff.time.attrs['c_grid_axis_shift']    # remove attribute from DataArray
G_total_Slt = sSALT_diff/delta_t

# Total tendency (deg C/s)
sTHETA_diff = sTHETA.diff('time_snp')
#sTHETA_diff = sTHETA_diff.rename({'time':'time_snp'})
#del sTHETA_diff.time.attrs['c_grid_axis_shift']    # remove attribute from DataArray
G_total_TH = sTHETA_diff/delta_t


################################################################################################
## Calculate tendency due to advective convergence
################################################################################################
### Horizontal convergence of advective heat/salt flux

# Set fluxes on land to zero (instead of NaN)
ds['ADVx_SLT'] = ds.ADVx_SLT.where(ecco_grid.hFacW.expand_dims(dim={"time_snp": len(ds.time_snp)}).values > 0,0)
ds['ADVy_SLT'] = ds.ADVy_SLT.where(ecco_grid.hFacS.expand_dims(dim={"time_snp": len(ds.time_snp)}).values > 0,0)
ds['ADVx_TH'] = ds.ADVx_TH.where(ecco_grid.hFacW.expand_dims(dim={"time_snp": len(ds.time_snp)}).values > 0,0)
ds['ADVy_TH'] = ds.ADVy_TH.where(ecco_grid.hFacS.expand_dims(dim={"time_snp": len(ds.time_snp)}).values > 0,0)

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
ds['ADVr_SLT'] = ds.ADVr_SLT.where(ecco_grid.hFacC.expand_dims(dim={"time_snp": len(ds.time_snp)}).values > 0,0)
ds['ADVr_TH'] = ds.ADVr_TH.where(ecco_grid.hFacC.expand_dims(dim={"time_snp": len(ds.time_snp)}).values > 0,0)

# Need to make sure that sequence of dimensions are consistent
ADVr_TH = ds.ADVr_TH.transpose('time_snp','tile','k_l','j','i')

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
ds['DFxE_SLT'] = ds.DFxE_SLT.where(ecco_grid.hFacW.expand_dims(dim={"time_snp": len(ds.time_snp)}).values > 0,0)
ds['DFyE_SLT'] = ds.DFyE_SLT.where(ecco_grid.hFacS.expand_dims(dim={"time_snp": len(ds.time_snp)}).values > 0,0)
ds['DFxE_TH'] = ds.DFxE_TH.where(ecco_grid.hFacW.expand_dims(dim={"time_snp": len(ds.time_snp)}).values > 0,0)
ds['DFyE_TH'] = ds.DFyE_TH.where(ecco_grid.hFacS.expand_dims(dim={"time_snp": len(ds.time_snp)}).values > 0,0)

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
ds['DFrE_TH'] = ds.DFrE_TH.where(ecco_grid.hFacC.expand_dims(dim={"time_snp": len(ds.time_snp)}).values > 0,0)
ds['DFrI_TH'] = ds.DFrI_TH.where(ecco_grid.hFacC.expand_dims(dim={"time_snp": len(ds.time_snp)}).values > 0,0)
ds['DFrE_SLT'] = ds.DFrE_SLT.where(ecco_grid.hFacC.expand_dims(dim={"time_snp": len(ds.time_snp)}).values > 0,0)
ds['DFrI_SLT'] = ds.DFrI_SLT.where(ecco_grid.hFacC.expand_dims(dim={"time_snp": len(ds.time_snp)}).values > 0,0)

# Load monthly averages of vertical diffusive fluxes
DFrE_TH = ds.DFrE_TH.transpose('time_snp','tile','k_l','j','i')
DFrI_TH = ds.DFrI_TH.transpose('time_snp','tile','k_l','j','i')

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
forcH_subsurf = ((q1*(mskC==1)-q2*(mskC.shift(k=-1)==1))*ds.oceQsw).transpose('time_snp','tile','k','j','i')

# Surface heat flux (W/m^2)
forcH_surf = ((ds.TFLUX - (1-(q1[0]-q2[0]))*ds.oceQsw)\
              *mskC[0]).transpose('time_snp','tile','j','i').assign_coords(k=0).expand_dims('k')
    
# Full-depth sea surface forcing (W/m^2)
forcH = xr.concat([forcH_surf,forcH_subsurf[:,:,1:]], dim='k').transpose('time_snp','tile','k','j','i')

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

salt = ds.SALT_snp
temp = ds.THETA_snp


################################################################################################
## Prep to save
################################################################################################

varnames = ['G_total_TH','adv_hConvHx','adv_hConvHy','adv_vConvH','dif_hConvHx','dif_hConvHy','dif_vConvH','G_forcing_TH',
            'G_total_Slt','adv_hConvSx','adv_hConvSy','adv_vConvS','dif_hConvSx','dif_hConvSy','dif_vConvS','G_forcing_Slt','salt','temp']


ds2 = xr.Dataset(data_vars={})
for varname in varnames:
    ds2[varname] = globals()[varname].chunk(chunks={'time_snp':1,'tile':1,'k':50,'j':90,'i':90})

# Add surface forcing (degC/s)
ds2['Qnet'] = ((forcH /(rhoconst*c_p))\
              /(ecco_grid.hFacC*ecco_grid.drF)).chunk(chunks={'time_snp':1,'tile':1,'k':50,'j':90,'i':90})

# Add shortwave penetrative flux (degC/s)
#Since we only are interested in the subsurface heat flxux we need to zero out the top cell
SWpen = ((forcH_subsurf /(rhoconst*c_p))/(ecco_grid.hFacC*ecco_grid.drF)).where(forcH_subsurf.k>0).fillna(0.)
ds2['SWpen'] = SWpen.where(ecco_grid.hFacC>0).chunk(chunks={'time_snp':1,'tile':1,'k':50,'j':90,'i':90})

ds2.time_snp.encoding = {}
ds2 = ds2.reset_coords(drop=True)


ds2['time_snp'] = time_snp[0:-1]


### Before this step, I pause and delete all other global variables in memory. It seems to help
### prevent the kernel from crashing

# with ProgressBar():
#     ds2.sel(time_snp=slice('2017-01-01','2017-12-31')).to_netcdf('/Users/joegradone/SynologyDrive/Drive/Rutgers/Research/data/ECCO/Evaluated_Budgets/eccov4r4_budg_heat_salt_2005_2017/eccov4r4_budg_heat_salt_2017.nc')

