"""Download iri_ecmwf netcdf files 

Example usage:
    for i in {1..50}
    do
        # python src/data/download/download_iri_ecmwf.py -gr us1_5 -fd 20150514-20220103 -v -se -sr $i -wv precip
        src/batch/batch_python.sh -m 1 --cores 1 --hours 2 src/data/download/download_iri_ecmwf.py -gr us1_5 -fd 20150514-20220103 -v -se -sr $i -wv precip
    done
"""
import subprocess
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse
import json
import pathlib
import sys
import time
from datetime import datetime
import gcsfs
import os
import xarray as xr

from abc_s2s.utils.data_util_old import df_contains_multiple_dates, df_contains_nas, is_valid_forecast_date, get_grid
from abc_s2s.utils.general_util import (
    download_url,
    dt_to_string,
    get_dates,
    get_folder,
    print_error,
    print_ok,
    print_info,
    print_warning,
    set_path_permission,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--weather_variable",
    "-wv",
    default="tmp2m",
    choices=["tmp2m", "precip"],
    help="Name of weather variable to download",
)
parser.add_argument(
    "--lead_times",
    "-lt",
    default="all",
    choices=["all", "34w", "56w"],
    help="Whether to include all aggregated leads, only the lead for the exact 34w period (i.e., "
    "starting at 15d ahead), or lead for exact 56w period (i.e., starting at 29d ahead)",
)
parser.add_argument(
    "--control_forecast",
    "-cf",
    action="store_true",
    help="If true, download just the control run; if false, download perturbed runs",
)
parser.add_argument(
    "--forecast_type",
    "-ft",
    default="forecast",
    choices=["forecast", "reforecast"],
    help="Whether to download forecast or reforecast (i.e., the hindcasts for the last 20 years)",
)
parser.add_argument(
    "--forecast_dates",
    "-fd",
    default="20200101",
    help="Dates to download data over; format can be '20200101-20200304', '2020', '202001', '20200104'",
)
parser.add_argument(
    "--geographic_region",
    "-gr",
    default="global1_5",
    choices=["global1_5", "us1_0", "us1_5"],
    help="Geographic region to download over",
)
parser.add_argument(
    "--verbose", "-v", action="store_true", help="Print verbose output",
)
parser.add_argument(
    "--check_file_integrity",
    "-cfi",
    action="store_true",
    help="Perform basic checks on downloaded file; e.g., each file only contains one forecast date, no nans",
)
parser.add_argument(
    "--skip_existing",
    "-se",
    action="store_true",
    help="If true, skips downloading data if resulting file already exists",
)
parser.add_argument(
    "--average_runs",
    "-ar",
    action="store_true",
    help="If true, download average of perturbed models",
)
parser.add_argument(
    "--single_run",
    "-sr",
    type=int,
    default=0,
    choices=range(51),
    help="If > 0, will download a single perturbed model run corresponding to that index",
)
parser.add_argument(
    "--keep_local",
    "-kl",
    action="store_true",
    help="If true, keeps local files after uploading to cloud storage.",
)
args = parser.parse_args()

local_storage = os.getenv('STORAGE_LOCAL')
cloud_storage = os.getenv('STORAGE_CLOUD')
if local_storage is None:
    local_storage = pathlib.Path("/Users/avocet/content/abc_s2s/")
if cloud_storage is None:
    cloud_storage = "gs://sheerwater-datalake"

# Open gcloud bucket
fs = gcsfs.GCSFileSystem(
    project='sheerwater', token='google_default')

# For each lead start time, average or sum over this number of days.
ACCUMULATE_LEADS_PERIOD = 14

weather_variable_names_on_server = {
    "tmp2m": "2m_above_ground/.2t",
    "precip": "sfc_precip/.tp",
}
weather_variable_name_on_server = weather_variable_names_on_server[args.weather_variable]
forecast_type = args.forecast_type
forecast_runs = "control" if args.control_forecast else "perturbed"
leads_id = "LA" if args.weather_variable == "tmp2m" else "L"
lead_shift = "14" if args.weather_variable == "tmp2m" else "13"
average_model_runs_url = "%5BM%5Daverage/" if args.average_runs else ""
single_model_run_url = f"M/%28{args.single_run}%29VALUES/" if args.single_run > 0 else ""

longitudes, latitudes, grid_size = get_grid(args.geographic_region)
restrict_latitudes_url = f"Y/{latitudes[0]}/{grid_size}/{latitudes[1]}/GRID/"
restrict_longitudes_url = f"X/{longitudes[0]}/{grid_size}/{longitudes[1]}/GRID/"


if args.weather_variable == "tmp2m":
    # Average temperature over the 2-week period, convert from Kelvin to Celsius.
    accumulate_leads_url = (
        f"{leads_id}/{ACCUMULATE_LEADS_PERIOD}/"
        f"runningAverage/{leads_id}/-{int(ACCUMULATE_LEADS_PERIOD/2)}/shiftGRID/"
    )
    convert_units_url = "(Celsius_scale)/unitconvert/"
else:
    # Precipitation is given as accumulated over lead time, so the data must be
    # subtracted from a lead-shifted version; this is done directly in the URL instead of here.
    accumulate_leads_url = ""
    convert_units_url = ""

if args.lead_times == "34w":
    restrict_leads_url = f"{leads_id}/%2815.0%29VALUES/"
elif args.lead_times == "56w":
    restrict_leads_url = f"{leads_id}/%2829.0%29VALUES/"
else:
    restrict_leads_url = ""


with open(pathlib.Path("/Users/avocet/content/abc_s2s/credentials.json"), "r") as credentials_file:
    credentials = json.load(credentials_file)
    ecmwf_key = credentials["ecmwf_key"]

if args.verbose:
    print_ok("ecmwf", bold=True)

perturbed_string = 'pf' if args.single_run == 0 else f'pf{args.single_run}'
folder_name = get_folder(
    f"data/forecast/iri_ecmwf/"
    f"{args.weather_variable}-{args.lead_times}-"
    f"{args.geographic_region}-{'cf' if args.control_forecast else perturbed_string}-{args.forecast_type}"
)


def download_and_write(forecast_date):
    """Download and write file to local storage and cloud storage."""
    file_path = local_storage / folder_name / \
        f"{dt_to_string(forecast_date)}.nc"

    cloud_path = f"{cloud_storage}/{folder_name}/{dt_to_string(forecast_date)}.nc"

    day, month, year = datetime.strftime(forecast_date, "%d,%b,%Y").split(",")

    if not is_valid_forecast_date("ecmwf", forecast_type, forecast_date):
        if args.verbose:
            print_warning(
                f"Skipping: {day} {month} {year} (not a valid {forecast_type} date).", skip_line_before=False,
            )
        return

    if fs.exists(cloud_path) and args.skip_existing:
        print_info(
            f"Skipping: {day} {month} {year} (file already exists).\n", verbose=args.verbose)
        return

    restrict_forecast_date_url = f"S/%28{day}%20{month}%20{year}%29VALUES/"

    t = time.time()
    if args.weather_variable == "tmp2m":
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

    r = download_url(URL, cookies={"__dlauth_id": ecmwf_key})
    if r.status_code == 200 and r.headers["Content-Type"] == "application/x-netcdf":
        print_info(f"Downloading: {day} {month} {year}.", verbose=args.verbose)

        with open(file_path, "wb") as f:
            f.write(r.content)

        print_info(
            f"-done (downloaded {sys.getsizeof(r.content)/1024:.2f} KB in {time.time() - t:.2f}s).\n",
            verbose=args.verbose,
        )

        if args.check_file_integrity:
            if df_contains_nas(file_path, weather_variable_name_on_server.split("/.")[1], how="any"):
                print_warning(
                    f"Warning: {day} {month} {year} contains nas in weather variable.",
                    skip_line_before=False,
                    skip_line_after=False,
                )
            if df_contains_multiple_dates(file_path, time_col="S"):
                print_warning(
                    f"Warning: {day} {month} {year} file contains multiple forecast dates.",
                    skip_line_before=False,
                    skip_line_after=False,
                )
    elif r.status_code == 404:
        print_warning(bold=True, verbose=args.verbose)
        print_info(
            f"Data for {day} {month} {year} is not available for model ecmwf.\n", verbose=args.verbose)
    else:
        print_error(bold=True)
        print_info(
            f"Unknown error occured when trying to download data for {day} {month} {year} for model ecmwf.\n")

    # Copy download files to google cloud storage and remove local files
    subprocess.run(
        f"gsutil cp {file_path} {cloud_path}", shell=True)

    if not args.keep_local:
        subprocess.run(
            f"rm {file_path}", shell=True)

if __name__ == '__main__':
    print("There are {} CPUs on this machine ".format(cpu_count()))
    tic = time.time()
    pool = Pool(cpu_count())

    target_dates = get_dates(args.forecast_dates)
    results = pool.map(download_and_write, target_dates)
    # results = pool.map(test, target_dates)
    pool.close()
    pool.join()
    print(f"Time taken: {time.time() - tic:.2f}s")