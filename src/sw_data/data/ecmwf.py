import dask
import sys
import time
import pandas as pd
from datetime import datetime
import os
import xarray as xr
import dateparser
from sheerwater_benchmarking.utils.caching import cacheable

from sw_data.utils.secrets import ecmwf_secret
from sw_data.utils.remote import dask_remote

from .utils import get_grid, print_ok, print_info, print_warning, print_error, \
    download_url, get_dates


@dask_remote
@cacheable(data_type='array',
           immutable_args=['time', 'variable', 'lead_time', 'forecast_type', 'run_type', 'grid'])
def single_iri_ecmwf(time, variable, lead_time, forecast_type,
                     run_type="average", grid="global1_5",
                     verbose=True):
    """ Fetches forecast data from the ECMWF IRI dataset.

    Args:
        time (str): The date to fetch data for (by day).
        variable (str): The weather variable to fetch.
        lead_time (str): The lead time of the forecast.
        forecast_type (str): The type of forecast to fetch. One of "forecast" or "hindcast".    
        run_type (str): The type of run to fetch. One of:
            - average: to download the averaged of the perturbed runs
            - control: to download the control forecast
            - [int 0-50]: to download a specific  perturbed run
        grid (str): The grid resolution to fetch the data at. One of:
            - global1_5: 1.5 degree global grid
        verbose (bool): Whether to print verbose output.
    """
    # For each lead start time, average or sum over this number of days.
    ACCUMULATE_LEADS_PERIOD = 14

    if variable == "tmp2m":
        weather_variable_name_on_server = "2m_above_ground/.2t"
    elif variable == "precip":
        weather_variable_name_on_server = "sfc_precip/.tp"
    else:
        raise ValueError("Invalid weather variable.")

    forecast_runs = "control" if run_type == "control" else "perturbed"
    leads_id = "LA" if variable == "tmp2m" else "L"
    average_model_runs_url = "%5BM%5Daverage/" if run_type == "average" else ""
    single_model_run_url = f"M/%28{
        run_type}%29VALUES/" if type(run_type) is int else ""

    longitudes, latitudes, grid_size = get_grid(grid)
    restrict_latitudes_url = f"Y/{latitudes[0]
                                  }/{grid_size}/{latitudes[1]}/GRID/"
    restrict_longitudes_url = f"X/{longitudes[0]
                                   }/{grid_size}/{longitudes[1]}/GRID/"

    if variable == "tmp2m":
        # Average temperature over the 2-week period, convert from Kelvin to Celsius.
        accumulate_leads_url = (
            f"{leads_id}/{ACCUMULATE_LEADS_PERIOD}/"
            f"runningAverage/{leads_id}/-{
                int(ACCUMULATE_LEADS_PERIOD/2)}/shiftGRID/"
        )
        convert_units_url = "(Celsius_scale)/unitconvert/"
    else:
        # Precipitation is given as accumulated over lead time, so the data must be
        # subtracted from a lead-shifted version; this is done directly in the URL instead of here.
        accumulate_leads_url = ""
        convert_units_url = ""

    if lead_time == "34w":
        restrict_leads_url = f"{leads_id}/%2815.0%29VALUES/"
    elif lead_time == "56w":
        restrict_leads_url = f"{leads_id}/%2829.0%29VALUES/"
    else:
        restrict_leads_url = ""

    ecmwf_key = ecmwf_secret()

    if verbose:
        print_ok("ecmwf", bold=True)

    DATETIME_FORMAT = "%Y-%m-%d"
    # date = dateparser.parse(day)
    date = datetime.strptime(time, DATETIME_FORMAT)
    day, month, year = datetime.strftime(date, "%d,%b,%Y").split(",")
    restrict_forecast_date_url = f"S/%28{day}%20{month}%20{year}%29VALUES/"

    if variable == "tmp2m":
        URL = (
            f"https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/"
            f".{forecast_type}/.{forecast_runs}/.{weather_variable_name_on_server}/"
            f"{average_model_runs_url}"
            f"{single_model_run_url}"
            f"{restrict_forecast_date_url}"
            f"{restrict_latitudes_url}"
            f"{restrict_longitudes_url}"
            f"{accumulate_leads_url}"
            f"{restrict_leads_url}"
            f"{convert_units_url}"
            f"data.nc"
        )
    else:
        # Since precipitation is accumulated over leads, shift the data and subtract from it the original data.
        # Drop the extra L_lag column (of constant value, since the shift is always of same size).
        base_url = "https://iridl.ldeo.columbia.edu/"
        model_url = "SOURCES/.ECMWF/.S2S/.ECMF/"
        modifiers_url = (
            f".{forecast_type}/.{forecast_runs}/.{weather_variable_name_on_server}/"
            f"{average_model_runs_url}"
            f"{single_model_run_url}"
            f"{restrict_forecast_date_url}"
            f"{restrict_latitudes_url}"
            f"{restrict_longitudes_url}"
        )
        URL = (
            f"{base_url}{model_url}{modifiers_url}"
            f"{leads_id}/{ACCUMULATE_LEADS_PERIOD}/shiftdata/"
            f"{model_url}{modifiers_url}sub/"
            f"{leads_id}_lag/removeGRID/{restrict_leads_url}data.nc"
        )

    os.makedirs('./temp', exist_ok=True)
    file = f"./temp/{variable}-{lead_time}-{grid}-{run_type}-{forecast_type}-{time}.nc"
    r = download_url(URL, cookies={"__dlauth_id": ecmwf_key})
    if r.status_code == 200 and r.headers["Content-Type"] == "application/x-netcdf":
        print_info(f"Downloading: {day} {month} {year}.", verbose=verbose)
        with open(file, "wb") as f:
            f.write(r.content)

        print_info(
            f"-done (downloaded {sys.getsizeof(r.content) / 1024:.2f} KB).\n",
            verbose=verbose,
        )
    elif r.status_code == 404:
        print_warning(bold=True, verbose=verbose)
        print_info(
            f"Data for {day} {month} {year} is not available for model ecmwf.\n", verbose=verbose)
        return None
    else:
        print_error(bold=True)
        print_info(
            f"Unknown error occured when trying to download data for {day} {month} {year} for model ecmwf.\n")
        return None

    # Read the data and return individual datasets
    if forecast_type == "forecast":
        ds = xr.open_dataset(file, engine="netcdf4")
        ds = ds.drop_vars("M", errors="ignore")
    else:
        ds = xr.open_dataset(file, decode_times=False)
        # Manuarlly decode the time variable
        ds['S'] = pd.to_datetime(
            ds['S'].values, unit="D", origin=pd.Timestamp("1960-01-01"))

        model_issuance_day = ds['S.day'].values[0]
        model_issuance_month = ds['S.month'].values[0]
        model_issuance_date_in_1960 = pd.Timestamp(
            f"1960-{model_issuance_month}-{model_issuance_day}")
        # While hdates refer to years (for which the model issued in ds["S"] is initialized), its values are given as
        # months until the middle of the year, so 6 months are subtracted to yield the beginning of the year.
        ds['hdate'] = pd.to_datetime(
            [model_issuance_date_in_1960 + pd.DateOffset(months=x-6) for x in ds['hdate'].values])

    os.remove(file)
    return ds


@dask_remote
# @cacheable(data_type='array',
#            immutable_args=['variable', 'lead_time', 'forecast_type', 'run_type', 'grid'])
def iri_ecmwf(start_time, end_time, variable, lead_time, forecast_type,
              run_type="average", grid="global1_5",
              verbose=True):
    """ Fetches forecast data from the ECMWF IRI dataset.

    Args:
        start_time (str): The start date to fetch data for.
        end_time (str): The end date to fetch.
        variable (str): The weather variable to fetch.
        lead_time (str): The lead time of the forecast.
        forecast_type (str): The type of forecast to fetch. One of "forecast" or "hindcast".    
        run_type (str): The type of run to fetch. One of:
            - average: to download the averaged of the perturbed runs
            - control: to download the control forecast
            - [int 0-50]: to download a specific  perturbed run
        grid (str): The grid resolution to fetch the data at. One of:
            - global1_5: 1.5 degree global grid
        verbose (bool): Whether to print verbose output.
    """
    # Read and combine all the data into an array
    target_dates = get_dates(start_time, end_time,
                             stride="day", return_string=True)

    datasets = []
    for date in target_dates:
        ds = dask.delayed(single_iri_ecmwf)(
            date, variable, lead_time, forecast_type, run_type, grid, verbose=True,
            filepath_only=True)
        datasets.append(ds)

    datasets = dask.compute(*datasets)
    ds = [d for d in datasets if d is not None]
    if len(ds) == 0:
        return None

    return
    # x = xr.open_mfdataset(ds, engine='zarr', concat_dim='S', combine="nested")
    # x = xr.open_mfdataset(ds, engine='zarr', combine="by_coords")
    # return x
