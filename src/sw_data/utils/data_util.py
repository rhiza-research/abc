import datetime
import pathlib
import pandas as pd

import xarray as xr
from abc_s2s.utils.general_util import string_to_dt
from abc_s2s.utils.experiments_util import get_climatology
import numpy as np
import copy

valid_forecast_dates = {
    "hindcast": {
        "30lcesm1": (string_to_dt("19990106"), string_to_dt("20151230"), "weekly"),
        "46lcesm1": (string_to_dt("19990106"), string_to_dt("20151230"), "weekly"),
        "gem": (string_to_dt("19950104"), string_to_dt("20141228"), "daily"),
        "geps5": (string_to_dt("19980103"), string_to_dt("20171227"), "daily"),
        "geps6": (string_to_dt("19980103"), string_to_dt("20171227"), "daily"),
        "geps7": (string_to_dt("20010101"), string_to_dt("20201231"), "daily"),
        "gefs": (string_to_dt("19990106"), string_to_dt("20161228"), "weekly"),
        "gefsv12": (string_to_dt("19890104"), string_to_dt("20190828"), "weekly"),
        "fimr1p1": (string_to_dt("19990106"), string_to_dt("20170628"), "weekly"),
        "geos_v2p1": (string_to_dt("19990101"), string_to_dt("20161227"), "daily"),
        "cfsv2": (string_to_dt("19990101"), string_to_dt("20170930"), "daily"),
        "nesm": (string_to_dt("19990102"), string_to_dt("20161231"), "daily"),
        "ccsm4": (string_to_dt("19990107"), string_to_dt("20161231"), "daily"),
        "ecmwf": (string_to_dt("20150514"), datetime.datetime.today(), "monday/thursday"),
    },
    "forecast": {
        "gem": (string_to_dt("20170727"), string_to_dt("20180913"), "daily"),
        "geps5": (string_to_dt("20180920"), string_to_dt("20190620"), "daily"),
        "geps6": (string_to_dt("20190627"), datetime.datetime.today(), "daily"),
        "geps7": (string_to_dt("20211202"), datetime.datetime.today(), "daily"),
        "gefs": (string_to_dt("20170630"), datetime.datetime.today(), "daily"),
        "gefsv12": (string_to_dt("20200930"), datetime.datetime.today(), "daily"),
        "fimr1p1": (string_to_dt("20170802"), datetime.datetime.today(), "daily"),
        "geos_v2p1": (string_to_dt("20170725"), datetime.datetime.today(), "daily"),
        "cfsv2": (string_to_dt("20171015"), datetime.datetime.today(), "daily"),
        "nesm": (string_to_dt("20170805"), datetime.datetime.today(), "daily"),
        "ccsm4": (string_to_dt("20170625"), datetime.datetime.today(), "weekly"),
        "ecmwf": (string_to_dt("20150514"), datetime.datetime.today(), "monday/thursday"),
    },
}

valid_measurement_dates = {
    "temp": (string_to_dt("19790101"), datetime.datetime.today(), "daily")
}


def generate_dates_in_between(first_date, last_date, date_frequency):
    if date_frequency == "monday/thursday":
        dates = [
            date
            for date in generate_dates_in_between(first_date, last_date, "daily")
            if date.strftime("%A") in ["Monday", "Thursday"]
        ]
        return dates
    else:
        frequency_to_int = {"daily": 1, "weekly": 7}
        dates = [
            first_date +
            datetime.timedelta(days=x * frequency_to_int[date_frequency])
            for x in range(0, int((last_date - first_date).days / (frequency_to_int[date_frequency])) + 1,)
        ]
        return dates


def is_valid_forecast_date(model, forecast_type, forecast_date):
    assert isinstance(forecast_date, datetime.datetime)

    # Treat ecmwf reforecast as a hindcast
    if model == "ecmwf" and forecast_type == "reforecast":
        forecast_type = "hindcast"
    try:
        return forecast_date in generate_dates_in_between(*valid_forecast_dates[forecast_type][model])
    except KeyError:
        return False


def is_valid_measurement_date(measurement, measurement_date):
    assert isinstance(measurement_date, datetime.datetime)

    try:
        return measurement_date in generate_dates_in_between(*valid_measurement_dates[measurement])
    except KeyError:
        return False


def get_grid(region_id):
    if region_id == "global1_5":
        longitudes = ["0", "358.5"]
        latitudes = ["-90.0", "90.0"]
        grid_size = "1.5"
    elif region_id == "global0_5":
        longitudes = ["0.25", "359.75"]
        latitudes = ["-89.75", "89.75"]
        grid_size = "0.5"
    elif region_id == "us1_0":
        longitudes = ["-125.0", "-67.0"]
        latitudes = ["25.0", "50.0"]
        grid_size = "1.0"
    elif region_id == "us1_5":
        longitudes = ["-123", "-67.5"]
        latitudes = ["25.5", "48"]
        grid_size = "1.5"
    else:
        raise NotImplementedError(
            "Only grids global1_5, us1_0 and us1_5 have been implemented.")
    return longitudes, latitudes, grid_size


def get_forecast_type(model, forecast_date):
    assert model != "ecmwf", "For ecmwf, forecast_type cannot be inferred by date; must be done explicitly."

    if forecast_date < valid_forecast_dates["hindcast"][model][1]:
        forecast_type = "hindcast"
    else:
        forecast_type = "forecast"
    return forecast_type


def df_is_all_nas(file_path, column_name):
    try:
        df = xr.open_dataset(file_path).to_dataframe().reset_index()
    except ValueError:
        df = open_ecmwf_reforecast_dataset(
            file_path).to_dataframe().reset_index()
    return df.isna()[column_name].all()


def df_contains_nas(file_path, column_name, how="any"):
    try:
        df = xr.open_dataset(file_path).to_dataframe().reset_index()
    except ValueError:
        df = open_ecmwf_reforecast_dataset(
            file_path).to_dataframe().reset_index()
    nas_in_column = df.isna()[column_name]
    if how == "all":
        return nas_in_column.all()
    elif how == "any":
        return nas_in_column.any()
    else:
        raise NotImplementedError("Flag 'how' must receive 'any' or 'all'.")


def df_contains_multiple_dates(file_path, time_col="S"):
    try:
        df = xr.open_dataset(file_path).to_dataframe().reset_index()
    except ValueError:
        df = open_ecmwf_reforecast_dataset(
            file_path).to_dataframe().reset_index()
    return len(df[time_col].unique()) > 1


def open_ecmwf_reforecast_dataset(file_path):
    """Opens an ecmwf reforecast nc file. It is not possible to use xr.open_dataset() because the encoding
    for the hindcast date (hdate) is given as months since 1960-01-01, which is not standard for xarray.
    """

    ds = xr.open_dataset(file_path, decode_times=False)
    ds['S'] = pd.to_datetime(ds['S'].values, unit="D",
                             origin=pd.Timestamp("1960-01-01"))

    model_issuance_day = ds['S.day'].values[0]
    model_issuance_month = ds['S.month'].values[0]
    model_issuance_date_in_1960 = pd.Timestamp(
        f"1960-{model_issuance_month}-{model_issuance_day}")
    # While hdates refer to years (for which the model issued in ds["S"] is initialized), its values are given as
    # months until the middle of the year, so 6 months are subtracted to yield the beginning of the year.
    ds['hdate'] = pd.to_datetime(
        [model_issuance_date_in_1960 + pd.DateOffset(months=x-6) for x in ds['hdate'].values])

    return ds


def get_subx_dataframe_from_nc_file(input_nc_path, model, args):
    df = xr.open_dataset(input_nc_path).to_dataframe().reset_index()
    df = df.set_index(["S", "X", "Y"]).pivot(columns="L").reset_index()

    base_cols = ["start_date", "lon", "lat"]
    if args.lead_times == "34w":
        forecast_cols = [f"iri_{model}_{args.weather_variable}-15.5d"]
    elif args.lead_times == "56w":
        forecast_cols = [f"iri_{model}_{args.weather_variable}-29.5d"]
    else:
        forecast_cols = [
            f"iri_{model}_{args.weather_variable}-{x}.5d" for x in range(0, df.shape[1] - len(base_cols))]

    df.columns = base_cols + forecast_cols
    df.start_date = df.start_date.dt.normalize()
    df.lon = df.lon + 360

    return df


def get_ecmwf_dataframe_from_nc_file(input_nc_path, args):
    leads_id = "LA" if args.weather_variable == "tmp2m" else "L"

    if args.forecast_type == "forecast":
        df = xr.open_dataset(input_nc_path).to_dataframe().reset_index()
        # Drop the "M" column if it exists
        df = df.drop("M", axis=1, errors="ignore").set_index(
            ["S", "X", "Y"]).pivot(columns=leads_id).reset_index()

    else:
        df = open_ecmwf_reforecast_dataset(
            input_nc_path).to_dataframe().reset_index()

        weather_variable_names_on_file = {"tmp2m": "2t", "precip": "tp"}
        dates_with_na = df["hdate"][df[weather_variable_names_on_file[args.weather_variable]].isnull(
        )].unique()
        df = df[~df["hdate"].isin(dates_with_na)]
        df = df.set_index(["S", "hdate", "X", "Y"]).pivot(
            columns=leads_id).reset_index()

    if args.forecast_type == "forecast":
        base_cols = ["start_date", "lon", "lat"]
    else:
        base_cols = ["model_issuance_date", "start_date", "lon", "lat"]
    if args.lead_times == "34w":
        forecast_cols = [f"iri_ecmwf_{args.weather_variable}-15.5d"]
    elif args.lead_times == "56w":
        forecast_cols = [f"iri_ecmwf_{args.weather_variable}-29.5d"]
    else:
        forecast_cols = [
            f"iri_ecmwf_{args.weather_variable}-{x}.5d" for x in range(0, df.shape[1] - len(base_cols))]
    df.columns = base_cols + forecast_cols

    df.lon = df.lon + 360

    return df


def get_temp_dataframe_from_nc_file(input_nc_path, args):
    ds = xr.open_dataset(input_nc_path)
    ds["X"] = ds["X"] + 360
    ds["T"] = pd.to_datetime(ds["T"].values, unit="D", origin="julian")

    df = ds.to_dataframe().reorder_levels(["T", "X", "Y"])
    df.index.rename(["start_date", "lon", "lat"], inplace=True)
    df.columns = ["temp"]

    return df


def load_mask_dataframe(geographic_region):
    if geographic_region == "us1_0":
        mask_file = pathlib.Path("src/setup/us_mask.nc")
    elif geographic_region == "us1_5":
        mask_file = pathlib.Path("data/masks/us_1_5_mask.nc")
    elif geographic_region == "global1_5":
        return None
    elif geographic_region == "global0_5":
        return None
    else:
        raise NotImplementedError(
            "Please specify which mask file to use with this geographic region.")

    ds = xr.open_dataset(mask_file)
    df = ds.to_dataframe().reset_index()
    df = df[df["mask"] == 1].drop("mask", axis=1)
    df.lon = df.lon + 360

    return df


def apply_mask_to_dataframe(df, geographic_region):
    mask_df = load_mask_dataframe(geographic_region)
    if mask_df is not None:
        df = pd.merge(df, mask_df, on=['lat', 'lon'], how='inner')
    return df


def get_ecmwf_indicator_dataframe_from_nc_file(input_nc_path, df_clim_terciles,
                                               weather_variable,
                                               forecast_type,
                                               lead_times,
                                               get_indicators):
    leads_id = "LA" if weather_variable == "tmp2m" else "L"

    # Set control and perturbed file names

    input_nc_path_control = input_nc_path if "-cf-" in str(
        input_nc_path) else pathlib.Path(str(input_nc_path).replace('-pf-', '-cf-'))
    input_nc_path_perturbed = input_nc_path if "-pf-" in str(
        input_nc_path) else pathlib.Path(str(input_nc_path).replace('-cf-', '-pf-'))

    if forecast_type == "forecast":
        # Load the control forecast
        df_control = xr.open_dataset(
            input_nc_path_control).to_dataframe().reset_index()
        df_control['M'] = 0
        # Load the perturbed forecast
        df = xr.open_dataset(
            input_nc_path_perturbed).to_dataframe().reset_index()
        # Append the perturbed forecasts to the control forecast
        df = df_control.append(df, sort=True)
        # Pivot dataframe
        df = df.set_index(["M", "S", "X", "Y"]).pivot(
            columns=leads_id).reset_index()

        # select relevant base columns, model_issuance_date depends on model and is irrelevant when avering over all models
        base_cols = ["model", "start_date", "lon", "lat"]
    else:
        # Load the control forecast
        df_control = open_ecmwf_reforecast_dataset(
            input_nc_path_control).to_dataframe().reset_index()
        df_control['M'] = 0
        # Load the perturbed forecasts
        df = open_ecmwf_reforecast_dataset(
            input_nc_path_perturbed).to_dataframe().reset_index()
        # Append the perturbed forecasts to the control forecast
        df = df_control.append(df, sort=True)
        # Pivot dataframe
        weather_variable_names_on_file = {"tmp2m": "2t", "precip": "tp"}
        dates_with_na = df["hdate"][df[weather_variable_names_on_file[weather_variable]].isnull(
        )].unique()
        df = df[~df["hdate"].isin(dates_with_na)]
        df = df.set_index(["M", "S", "hdate", "X", "Y"]).pivot(
            columns=leads_id).reset_index()

        # select relevant base columns, model_issuance_date depends on model and is irrelevant when avering over all models
        base_cols = ["model", "model_issuance_date",
                     "start_date", "lon", "lat"]

    # Select relevant lags
    if lead_times == "34w":
        forecast_cols = [f"iri_ecmwf_{weather_variable}-15.5d"]
    elif lead_times == "56w":
        forecast_cols = [f"iri_ecmwf_{weather_variable}-29.5d"]
    else:
        forecast_cols = [
            f"iri_ecmwf_{weather_variable}-{x}.5d" for x in range(0, df.shape[1] - len(base_cols))]
    df.columns = base_cols + forecast_cols

    # df.lon = df.lon + 360

    # Get number of ensemble members
    num_models = len(df["model"].unique())

    # Get lat lons
    lat_lon_clim = [(i_lat, i_lon) for i_lat, i_lon in zip(
        df_clim_terciles['lat'], df_clim_terciles['lon'])]
    lat_lon_iri = df.groupby(['lat', 'lon']).size().reset_index()
    lat_lon_iri = [(i_lat, i_lon) for i_lat, i_lon in zip(
        lat_lon_iri['lat'], lat_lon_iri['lon'])]
    lat_lons = [l for l in lat_lon_iri if l in lat_lon_clim]
    df_clim_terciles = df_clim_terciles.set_index(['lat', 'lon'])

    df_p = copy.copy(df)
    # join with terciles values
    df_p = df.join(df_clim_terciles, on=["lat", "lon"])
    # drop lat lons without climatology terciles
    df_p = df_p.dropna(axis=0, subset=df_clim_terciles.columns)
    df_p = df_p.drop("model", axis=1)  # drop the model column
    df_p = df_p.drop(df_clim_terciles.columns, axis=1)

    if forecast_type == "forecast":
        groupby_list = ["lat", "lon", "start_date"]
    else:
        groupby_list = ["lat", "lon", "model_issuance_date", "start_date"]

    ret = {}
    # TODO: should be able to do this as a groupby and aggregate to improve
    # efficiency dramatically
    for grp, df_i in df_p.groupby(groupby_list):
        i_lat = grp[0]
        i_lon = grp[1]

        # Get tercile bin thresholds for specific grid cell
        t1 = df_clim_terciles.loc[i_lat, i_lon][f'{weather_variable}_t1']
        t2 = df_clim_terciles.loc[i_lat, i_lon][f'{weather_variable}_t2']

        if get_indicators == "p1":
            ret[grp] = df_i[forecast_cols].le(
                t1).sum(axis=0).div(df_i.shape[0])
        else:
            ret[grp] = df_i[forecast_cols].gt(
                t2).sum(axis=0).div(df_i.shape[0])

    df_ret = pd.DataFrame.from_dict(ret, orient="index")
    df_ret.index.set_names(names=groupby_list, inplace=True)
    df_ret = df_ret.reset_index()
    return df_ret


def get_clim_terciles(gt_id):
    clim = get_climatology(gt_id)
    gt_var = 'precip' if 'precip' in gt_id else 'tmp2m'
    df = clim.groupby(['lat', 'lon'])[[gt_var]].agg(
        lambda g: np.percentile(g, 33.3))
    df = df.rename(columns={gt_var: f'{gt_var}_t1'})
    df[f"{gt_var}_t2"] = clim.groupby(['lat', 'lon'])[[gt_var]].agg(
        lambda g: np.percentile(g, 66.6))
    return df.reset_index()
