"""Convert iri_ecmwf netcdf files into a pandas dataframes 

Example usages:
    python src/data/pandify/pandify_iri_ecmwf.py -wv tmp2m -ft forecast -fd 20150514 -gr us1_0 -cn -gi p1 -v 
  
    python src/data/pandify/pandify_iri_ecmwf.py -gr us1_5 -fd 20150514-20220103 -v -cn -sr 1 -wv precip 
    for i in {1..50}
    do
        src/batch/batch_python.sh -m 1 --cores 1 --hours 0 --minutes 5 src/data/pandify/pandify_iri_ecmwf.py -gr us1_5 -fd 20150514-20220103 -v -cn -sr $i -wv tmp2m 
    done
  
Named args:
    --weather_variable (-wv): name of weather variable to download.
    --lead_times (-lt): whether to include all aggregated leads, only the lead for the exact 34w period (i.e., starting at 15d ahead), or lead for exact 56w period (i.e., starting at 29d ahead).
    --control_forecast (-cf): if true, download just the control forecasts; if false, use average of perturbed models.
    --forecast_type (-ft): whether to download forecast or reforecast (i.e., the hindcasts for the last 20 years).
    --forecast_dates (-fd): dates to download data over; format can be '20200101-20200304', '2020', '202001', '20200104'.
    --geographic_region (-gr): geographic region to download over.
    --create_new (-cn): if true, create new dataframe instead of appending to existing one.
    --raise_error_if_missing_forecast (-eim): if true, raise error as soon as a requested forecast doesn't exist; if false, skip it.
    --download_if_missing_forecast (-dim): if true (and --eim is not on) download requested forecast doesn't exist; if false, skip it.
    --get_indicators (-gi): get average of the indicator function of belonging to the 1st and 3rd terciles over all ensemble members
    --verbose (-v): print verbose output.
    --single_run (-sr): if > 0, will process a single perturbed model run corresponding to that index.
"""

import argparse
import pathlib
import os
import subprocess
import sys
import time
import pandas as pd
import gcsfs
from sw_data.utils.data_util import *
from sw_data.utils.general_util import dt_to_string, get_dates, print_error, print_ok, print_info, print_warning, set_path_permission

parser = argparse.ArgumentParser()
parser.add_argument("--weather_variable", "-wv", default="tmp2m",
                    choices=["tmp2m", "precip"], help="Name of weather variable to download")
parser.add_argument("--lead_times", "-lt", default="all",
                    choices=["all", "34w", "56w"])
parser.add_argument("--control_forecast", "-cf", action="store_true")
parser.add_argument("--forecast_type", "-ft",
                    default="forecast", choices=["forecast", "reforecast"])
parser.add_argument("--forecast_dates", "-fd", default="20200101")
parser.add_argument("--geographic_region", "-gr",
                    default="global1_5", choices=["global1_5", "us1_0", "us1_5"])
parser.add_argument("--create_new", "-cn", action="store_true")
parser.add_argument("--raise_error_if_missing_forecast",
                    "-eim", action="store_true")
parser.add_argument("--download_if_missing_forecast",
                    "-dim", action="store_true")
parser.add_argument("--get_indicators", "-gi",
                    default="", choices=["p1", "p3"])
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument(
    "--single_run",
    "-sr",
    type=int,
    default=0,
    choices=range(51),
    help="If > 0, will process a single perturbed model run corresponding to that index",
)
args = parser.parse_args()

weather_variable = args.weather_variable
lead_times = args.lead_times
control_forecast = args.control_forecast
forecast_type = args.forecast_type
forecast_dates = args.forecast_dates
geographic_region = args.geographic_region
create_new = args.create_new
raise_error_if_missing_forecast = args.raise_error_if_missing_forecast
download_if_missing_forecast = args.download_if_missing_forecast
get_indicators = args.get_indicators
verbose = args.verbose
single_run = args.single_run

local_storage = os.getenv('STORAGE_LOCAL')
cloud_storage = os.getenv('STORAGE_CLOUD')

# Connect to cloud buckets
fs = gcsfs.GCSFileSystem(project='sheerwater', token='google_default')

weather_variable_name_on_server = "2t" if weather_variable == "tmp2m" else "tp"

print_ok("ecmwf", verbose=verbose, bold=True)

# Create initial list of dataframe to concat
perturbed_string = 'pf' if single_run == 0 else f'pf{single_run}'
if create_new:
    existing_df = None
    existing_forecast_dates = []
else:
    existing_filepath = f"{cloud_storage}/data/dataframes/iri-ecmwf-{weather_variable}-{
        lead_times}-{geographic_region}-{'cf' if control_forecast else perturbed_string}-{forecast_type}.zarr"
    try:
        gcsmap = fs.get_mapper(existing_filepath)
        existing_df = xr.open_dataset(gcsmap, engine="zarr")
    except FileNotFoundError as e:
        print_error(f"Could not find existing dataframe to which new data should be appended (i.e., {existing_filepath})."
                    f" If you intend to create it for the first time, please rerun this script with flag --create_new.", bold=True)
        exit()
    reference_date_col = "start_date" if forecast_type == "forecast" else "model_issuance_date"
    existing_forecast_dates = pd.to_datetime(
        existing_df[reference_date_col].unique())

dfs_to_concat = [existing_df]

# populate dataframe with data for each forecast date
for forecast_date in get_dates(forecast_dates):
    # Path to input file
    input_nc_path = f"{cloud_storage}/data/forecast/iri_ecmwf/{weather_variable}-{lead_times}-{geographic_region}-{
        'cf' if control_forecast else perturbed_string}-{forecast_type}/{dt_to_string(forecast_date)}.nc"

    if forecast_date in existing_forecast_dates:
        print_warning(f"Skipping: {dt_to_string(forecast_date)} (date already exists in dataframe).",
                      skip_line_before=False, verbose=verbose)
        continue
    elif not is_valid_forecast_date("ecmwf", forecast_type, forecast_date):
        print_warning(f"Skipping: {dt_to_string(forecast_date)} (invalid {forecast_type} date).",
                      skip_line_before=False, verbose=verbose,)
        continue
    elif not fs.exists(input_nc_path):
        if raise_error_if_missing_forecast:
            print_error(f"Missing: {dt_to_string(
                forecast_date)} (file does not exist).", bold=True)
            raise FileNotFoundError
        elif download_if_missing_forecast:
            print_info(f"Downloading: {dt_to_string(
                forecast_date)} (file does not exist)", verbose=verbose)
            t = time.time()
            call_args = ["python", pathlib.Path("src/data/download/download_iri_ecmwf.py"),
                         "--weather_variable", weather_variable,
                         "--lead_times", lead_times,
                         "--forecast_dates", dt_to_string(forecast_date),
                         "--geographic_region", geographic_region,
                         "--forecast_type", forecast_type,
                         "--single_run", single_run,
                         "--check_file_integrity"]
            if control_forecast:
                call_args.append("--control_forecast")
            subprocess.call(call_args)
            print_info(f"-done (in {time.time()-t:.2f}s)", verbose=verbose)
        else:
            print_warning(f"Skipping: {dt_to_string(
                forecast_date)} (file does not exist)", skip_line_before=False)
            continue

    fp = fs.open(input_nc_path)
    if df_contains_nas(fp, weather_variable_name_on_server, how="all", engine="h5netcdf"):
        print_warning(f"Skipping: {dt_to_string(
            forecast_date)} (weather variable is all nas)", skip_line_before=False)
        continue

    # date specific dataframe
    print_info(f"Loading: {dt_to_string(forecast_date)}", verbose=verbose)
    t = time.time()
    fp = fs.open(input_nc_path)
    df = xr.open_dataset(df, engine="h5netcdf")
    print_info(f"-done (in {time.time()-t:.2f}s)", verbose=verbose)

    print_info("Applying mask", verbose=verbose)
    t = time.time()
    df = apply_mask_to_dataframe(df, geographic_region)
    print_info(f"-done (in {time.time()-t:.2f}s)\n", verbose=verbose)

    dfs_to_concat.append(df)

print_info("Concatenating dataframes", verbose=verbose)
t = time.time()
df = xr.concat(dfs_to_concat, dim="S")
print_info(f"-done (in {time.time()-t:.2f}s)\n", verbose=verbose)

print_info("Sorting dataframe", verbose=verbose)
t = time.time()
df = df.sort_values(by=["start_date", "lon", "lat"])
print_info(f"-done (in {time.time()-t:.2f}s)\n", verbose=verbose)

output_filepath = f"{cloud_storage}/data/dataframes/iri-ecmwf-{weather_variable}-{lead_times}-{
    geographic_region}-{'cf' if control_forecast else perturbed_string}-{forecast_type}.zarr"

print_info(f"Saving dataframe", verbose=verbose)
df.to_hdf(output_filepath, mode="w", key="data")
print_info(f"Dataframe saved: {output_filepath} (size: ~{
           sys.getsizeof(df)/(2**20):.2f} MB)\n", verbose=verbose)
