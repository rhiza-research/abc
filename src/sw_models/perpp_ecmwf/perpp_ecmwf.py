#!/usr/bin/env python
# coding: utf-8
# %%

# # Persistence++ with ECMWF
# 
# Regress onto ECMWF forecasts, climatology, and lagged measurements

# %%


import os, sys
from sw_models.utils.notebook_util import isnotebook
if isnotebook():
    # Autoreload packages that are modified
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
else:
    from argparse import ArgumentParser

# Imports 
import numpy as np
import pandas as pd
from sklearn import *
import sys
import json
import subprocess
from datetime import datetime, timedelta
from functools import partial
from multiprocessing import cpu_count
from sw_models.utils.data_utils import get_measurement_variable, df_merge, shift_df
from sw_models.utils.general_util import printf, tic, toc
from sw_models.utils.experiments_util import (get_first_year, month_day_subset, get_ground_truth,
                                                        get_start_delta, clim_merge, get_forecast_delta)
from sw_models.utils.eval_util import get_target_dates, mean_rmse_to_score, save_metric
from sw_models.utils.fit_and_predict import apply_parallel
from sw_models.utils.models_util import (get_submodel_name, start_logger, log_params, get_forecast_filename,
                                                   save_forecasts)
from sw_models.utils.perpp_util import fit_and_predict, years_ago
# from subseasonal_data import data_loaders

import re

# %%


#
# Specify model parameters
#
model_name = "perpp_ecmwf"
if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon                                                                                  
    parser.add_argument('--target_dates', '-t', default="std_contest")
    
    # Number of years to use in training ("all" or integer)
    parser.add_argument('--train_years', '-y', default="all")
    
    # Number of month-day combinations on either side of the target combination 
    # to include when training
    # Set to 0 to include only target month-day combo
    # Set to "None" to include entire year
    parser.add_argument('--margin_in_days', '-m', default="None")
    
    # Which version of the ECMWF forecasts to use when training
    # Valid choices include cf (for control forecast), 
    # pf (for average perturbed forecast), ef (for control+perturbed ensemble),
    # or pf1, ..., pf50 for a single perturbed forecast
    parser.add_argument('--version', '-v', default="ef")

    args, opt = parser.parse_known_args()
    
    # Assign variables                                                                                                                                     
    gt_id = args.pos_vars[0] # "contest_precip" or "contest_tmp2m"                                                                            
    horizon = args.pos_vars[1] # "34w" or "56w"                                                                                        
    target_dates = args.target_dates
    train_years = args.train_years
    if train_years != "all":
        train_years = int(train_years)
    if args.margin_in_days == "None":
        margin_in_days = None
    else:
        margin_in_days = int(args.margin_in_days)
    version = args.version
            
else:
    # Otherwise, specify arguments interactively 
    gt_id = "us_tmp2m_1.5x1.5"
    horizon = "12w"
    target_dates = "std_ecmwf"
    train_years = "all"
    margin_in_days = None
    version = "pf1"

#
# Process model parameters
#


# %%


# Get list of target date objects
target_date_objs = pd.Series(get_target_dates(date_str=target_dates,horizon=horizon))

# Sort target_date_objs by day of week
target_date_objs = target_date_objs[target_date_objs.dt.weekday.argsort(kind='stable')]

# Identify measurement variable name
measurement_variable = get_measurement_variable(gt_id) # 'tmp2m' or 'precip'

# Column names for gt_col, clim_col and anom_col 
gt_col = measurement_variable
clim_col = measurement_variable+"_clim"
anom_col = get_measurement_variable(gt_id)+"_anom" # 'tmp2m_anom' or 'precip_anom'

# For a given target date, the last observable training date is target date - gt_delta
# as gt_delta is the gap between the start of the target date and the start of the
# last ground truth period that's fully observable at the time of forecast issuance
gt_delta = timedelta(days=get_start_delta(horizon, gt_id))


# %%


#
# Choose regression parameters
#
# Record standard settings of these parameters
base_col = "zeros"    
if (gt_id.endswith("tmp2m_1.5x1.5")) and (horizon == "12w"):
    ecmwf_col = 'iri_ecmwf_tmp2m'
    x_cols = [
    'tmp2m_shift15',
    'tmp2m_shift30',
    ecmwf_col,
    clim_col
    ] 
elif (gt_id.endswith("precip_1.5x1.5")) and (horizon == "12w"):
    ecmwf_col = 'iri_ecmwf_precip'
    x_cols = [
    'precip_shift15',
    'precip_shift30',
    ecmwf_col,
    clim_col
    ] 
elif (gt_id.endswith("tmp2m_1.5x1.5")) and (horizon == "34w"):
    ecmwf_col = 'iri_ecmwf_tmp2m'
    x_cols = [
    'tmp2m_shift29',
    'tmp2m_shift58',
    ecmwf_col,
    clim_col
    ] 
elif (gt_id.endswith("precip_1.5x1.5")) and (horizon == "34w"):
    ecmwf_col = 'iri_ecmwf_precip'
    x_cols = [
    'precip_shift29',
    'precip_shift58',
    ecmwf_col,
    clim_col
    ] 
elif (gt_id.endswith("tmp2m_1.5x1.5")) and (horizon == "56w"):
    ecmwf_col = 'iri_ecmwf_tmp2m'
    x_cols = [
    'tmp2m_shift43',
    'tmp2m_shift86',
    ecmwf_col,
    clim_col
    ] 
elif (gt_id.endswith("precip_1.5x1.5")) and (horizon == "56w"):
    ecmwf_col = 'iri_ecmwf_precip'
    x_cols = [
    'precip_shift43',
    'precip_shift86',
    ecmwf_col,
    clim_col
    ]
else:
    raise ValueError(f"{model_name} currently only supports US 1.5 x 1.5 gt_ids.")
group_by_cols = ['lat', 'lon']

# Record submodel names for perpp model
submodel_name = get_submodel_name(
    model_name, train_years=train_years, margin_in_days=margin_in_days, 
    version=version)

printf(f"Submodel name {submodel_name}")

if not isnotebook():
    # Save output to log file
    logger = start_logger(model=model_name,submodel=submodel_name,gt_id=gt_id,
                          horizon=horizon,target_dates=target_dates)
    # Store parameter values in log                                                                                                                        
    params_names = ['gt_id', 'horizon', 'target_dates',
                    'train_years', 'margin_in_days', 'version',
                    'base_col', 'x_cols', 'group_by_cols'
                   ]
    params_values = [eval(param) for param in params_names]
    log_params(params_names, params_values)


# %%
#
# Load ground truth data
#
printf("Loading ground truth data")
tic()
import pdb; pdb.set_trace()
gt = get_ground_truth(gt_id)[['lat','lon','start_date',gt_col]]
toc()


# %%
#
# Added shifted ground truth features
#
printf("Adding shifted ground truth features")
lld_data = gt
shifts = [int(re.search(r'\d+$', col).group()) for col in x_cols if col.startswith(gt_col+"_shift")]
tic()
for shift in shifts:
    gt_shift = shift_df(gt, shift)
    lld_data = df_merge(lld_data, gt_shift, how="right")
toc()

#
# Drop rows with empty pred_cols
#
pred_cols = x_cols+[base_col]
exclude_cols = set([clim_col, ecmwf_col, 'zeros']) 
lld_data = lld_data.dropna(subset=set(pred_cols) - exclude_cols)

# Add climatology
if clim_col in pred_cols:
    printf("Merging in climatology")
    tic()
    lld_data = clim_merge(lld_data, data_loaders.get_climatology(gt_id))
    toc()

# Add zeros
if 'zeros' in pred_cols:
    lld_data['zeros'] = 0


# %%


#
# Add ECMWF forecast and reforecast features
#

# The number of days between a target date and the associated ECMWF model issuance date 
ecmwf_delta = get_forecast_delta(horizon)

def get_ecmwf_features(forecast=True, version="ef", region="us"):
    """Returns ECMWF forecasts or reforecasts for a lead determined by
    the target horizon
    
    Args: 
      forecast: if True, load forecasts; otherwise load reforecasts
    """
    shift = ecmwf_delta
    if forecast:
        suffix = "forecast"
        cols = ['lat','lon']
    else:
        suffix = "reforecast"
        cols = [f'model_issuance_date_shift{shift}','lat','lon']
    printf(f"Preparing ECMWF {suffix}s with shift {shift} and lead {shift}...")
    # Load ECMWF forecast data by forecast_id of the form 'ecmwf-tmp2m-us1_5-ef-forecast'
    tic()
    if region in ['us', 'global']:
        data = data_loaders.get_forecast(f"ecmwf-{gt_col}-{region}1_5-{version}-{suffix}", shift=shift, sync=False);
    else:
        raise ValueError(f"{model_name} currently only supports US 1.5 x 1.5 and global 1.5 x 1.5 gt_ids.")
    toc()
    tic()
    # Column names take the form 'iri_ecmwf_tmp2m-14.5d_shift15'
    forecast_col = f'iri_ecmwf_{gt_col}-{shift}.5d_shift{shift}'
    data = data.loc[:, cols + ['start_date',forecast_col]].rename(
        columns={forecast_col: ecmwf_col})
    toc()
    if not forecast:
        data = data.rename(columns={f"model_issuance_date_shift{shift}": "model_issuance_date"})
    else:
        # For forecasts, the model_issuance date is the start_date minus the shift
        data['model_issuance_date'] = data['start_date'] - timedelta(shift)
    return data

# Concatenate ECMWF forecast and reforecast dataframes
ecmwf = get_ecmwf_features(True, version, region)
# Always use control-perturbed ensemble ("ef") for reforecasts
ecmwf = pd.concat([ecmwf, get_ecmwf_features(False, "ef", region)])

# Merge ecmwf features with lld_data
printf(f"Merging {ecmwf_col} with lld_data")
tic()
lld_data = pd.merge(lld_data, ecmwf, on=['lat','lon','start_date'])
toc()

printf(f"Stably sorting lld_data by model issuance date")
tic()
lld_data = lld_data.sort_values(by="model_issuance_date", kind='stable')
toc()


# %%


# specify regression model
fit_intercept = True
model = linear_model.LinearRegression(fit_intercept=fit_intercept)

# Form predictions for each grid point (in parallel) using train / test split
# and the selected model
prediction_func = partial(fit_and_predict, model=model)
num_cores = cpu_count()

# Store rmses
rmses = pd.Series(index=target_date_objs, dtype='float64')

# Restrict data to relevant columns and rows for which predictions can be made
relevant_cols = set(
    ['model_issuance_date','start_date','lat','lon',gt_col,base_col]+x_cols).intersection(lld_data.columns)
lld_data = lld_data[relevant_cols].dropna(subset=x_cols+[base_col])


# %%


for target_date_obj in target_date_objs:
    # Ignore datapoints with model_issuance dates > target_date - forecast_delta and
    # drop duplicate lat-lon-start_date combinations, preserving only the last occurence
    # (which corresponds to the datapoint with the latest remaining model issuance date 
    # as lld_data is sorted by model issuance date)
    sub_data = lld_data[
        lld_data.model_issuance_date <= 
        target_date_obj - timedelta(ecmwf_delta)].drop_duplicates(
        subset=['lat','lon','start_date'], keep='last')
    
    if not any(sub_data.start_date.isin([target_date_obj])):
        printf(f"warning: some features unavailable for target={target_date_obj}; skipping")
        continue    
        
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
    
    # Skip if forecast already produced for this target
    forecast_file = get_forecast_filename(
        model=model_name, submodel=submodel_name, 
        gt_id=gt_id, horizon=horizon, 
        target_date_str=target_date_str)
    
    if True and os.path.isfile(forecast_file):
        printf(f"prior forecast exists for target={target_date_obj}; loading")
        tic()
        preds = pd.read_hdf(forecast_file)
        
        # Add ground truth for later evaluation
        preds = pd.merge(preds, sub_data.loc[sub_data.start_date==target_date_obj,['lat','lon',gt_col]], 
                         on=['lat','lon'])
        
        preds.rename(columns={gt_col:'truth'}, inplace=True)
        toc()
    else:
        printf(f'target={target_date_str}')
        
        # Subset data based on margin
        if margin_in_days is not None:
            tic()
            sub_data = month_day_subset(sub_data, target_date_obj, margin_in_days)
            toc()
            
        # Find the last observable training date for this target
        last_train_date = target_date_obj - gt_delta 
        
        # Only train on train_years worth of data
        if train_years != "all":
            tic()
            sub_data = sub_data.loc[sub_data.start_date >= years_ago(last_train_date, train_years)]
            toc()
            
        tic()
        preds = apply_parallel(
            sub_data.groupby(group_by_cols),
            prediction_func, 
            num_cores=num_cores,
            gt_col=gt_col,
            x_cols=x_cols, 
            base_col=base_col, 
            last_train_date=last_train_date,
            test_dates=[target_date_obj])
        
        # Ensure raw precipitation predictions are never less than zero
        if gt_id.endswith("precip") or gt_id.endswith("precip_1.5x1.5"):
            tic()
            preds['pred'] = np.maximum(preds['pred'],0)
            toc()
            
        preds = preds.reset_index()
        
        if True:
            # Save prediction to file in standard format
            save_forecasts(preds.drop(columns=['truth']),
                model=model_name, submodel=submodel_name, 
                gt_id=gt_id, horizon=horizon, 
                target_date_str=target_date_str)
        toc()
    
    # Evaluate and store error
    rmse = np.sqrt(np.square(preds.pred - preds.truth).mean())
    rmses.loc[target_date_obj] = rmse
    print("-rmse: {}, score: {}".format(rmse, mean_rmse_to_score(rmse)))
    mean_rmse = rmses.mean()
    print("-mean rmse: {}, running score: {}".format(mean_rmse, mean_rmse_to_score(mean_rmse)))

if True:
    # Save rmses in standard format
    rmses = rmses.reset_index()
    rmses.columns = ['start_date','rmse']
    save_metric(rmses, model=model_name, submodel=submodel_name, gt_id=gt_id, horizon=horizon, target_dates=target_dates, metric="rmse")


# %%




