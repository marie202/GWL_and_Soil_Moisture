# --- Imports and setup ---

# seed
from numpy.random import seed
seed(1+347823)
import tensorflow as tf
tf.random.set_seed(1+63493)

# Standard libraries
import numpy as np
import os, glob
import pandas as pd
import datetime
from scipy import stats
import matplotlib.pyplot as plt
from uncertainties import unumpy
import geopandas as gpd
import random, shutil

# Keras imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import load_model

# Bayesian optimization
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition

# Scikit-learn for data prep
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import shap

# List available GPUs (for info/debug)
gpus = tf.config.experimental.list_physical_devices('GPU')

# Import project modules
from s1_data_preparation import *
from s2_model_utils import *

# Main function for Bayesian optimization (converts floats to ints for discrete params)
def bayesOpt_function(
    densesize, 
    windowsize, 
    batchsize, 
    filters):
    """
    Main function for Bayesian optimization.
    Converts float params to ints for discrete params.
    Returns R² as optimization metric

    inspired by Wunsch et al 2022 on github https://github.com/AndreasWunsch/Long-Term-GWL-Simulations
    """

    # Convert float params to int for model
    # basically means conversion to rectangular function
    densesize_int = int(densesize)
    windowsize_int = int(windowsize)
    batchsize_int = int(batchsize)
    filters_int = int(filters)

    
    return bayesOpt_function_with_discrete_params(
        densesize_int, 
        windowsize_int, 
        batchsize_int, 
        filters_int,
    )

# Actual function called by Bayesian optimization, builds and evaluates model
def bayesOpt_function_with_discrete_params(
    densesize_int, 
    windowsize_int, 
    batchsize_int, 
    filters_int,
    ):
    """
    Actual function called by Bayesian optimization, builds and evaluates model
    inspired and adapted from Wunsch et al 2022 on github https://github.com/AndreasWunsch/Long-Term-GWL-Simulations
    """
    # Ensure correct types for all params
    assert type(densesize_int) == int
    assert type(windowsize_int) == int
    assert type(batchsize_int) == int
    assert type(filters_int) == int


    # Set model/global settings for this run
    GLOBAL_SETTINGS = {
        'inimax': 3,  # number of initializations for uncertainty
        'batch_size': batchsize_int,
        'kernel_size': 3,  # must be odd
        'dense_size': densesize_int,
        'filters': filters_int,
        'window_size': windowsize_int,
        'clip_norm': True,
        'clip_value': 1,
        'epochs': 250,
        'learning_rate': 1e-5,
        'test_start': pd.to_datetime('2018-01-01', format='%Y-%m-%d'),
        'test_end': pd.to_datetime('2024-09-01', format='%Y-%m-%d'),
        'num_cnn_layers': 5,
        'lstm_units': [32, 16], #[32, 16],
    # 'model_dir_note': "L2_1e5_LSTM32_16_bester_run_new_NS_and_kf_filtered_new_scalerlogic_bayesopt"
    }
    # Features to use for training (last entry must be 'GWL')
    columns_to_keep = [
        'tas_3x3_mean',
        'pr_3x3_mean',
        'hurs_3x3_mean',
        #'soil_moisture_xy_general',
        'soil_mois_composite_3x3_mean_0-30',
        'soil_mois_composite_3x3_mean_0-60',
        'soil_mois_composite_3x3_mean_0-90',
        'elevation_msl',
        'MW_muGOK',
        'distance_to_waterwork_km',
        'kf_remapped',
        'GWL'
    ]
    print("Columns to train on: ", columns_to_keep)


    static_cols = [
        'elevation_msl',
        'MW_muGOK',
        'distance_to_waterwork_km',
        'kf_remapped',
        ] 
    # Input data directory (all CSVs)
    input_dir = "data_filtered_anthro/*.csv"
    # List of well IDs from filenames
    well_ids = [os.path.basename(file).split('_')[0] for file in glob.glob(input_dir)]#

    scaler_static, _ = scaler_statics_global(
        input_dir=input_dir,
        static_cols=static_cols,
        columns_to_keep=columns_to_keep
        )


    # Prepare data for this run
    # --- Use function pipeline for data preparation  ---
    X_train, Y_train, X_val, Y_val, X_opt, Y_opt, ScalerData_dict, ValData_dict, OptData_dict, TestData_dict = process_data_pipeline(
        input_dir=input_dir,
        columns_to_keep=columns_to_keep,
        static_cols=static_cols,
        GLOBAL_SETTINGS=GLOBAL_SETTINGS,
        scaler_static=scaler_static,
        target_column = "GWL"
    )

    # Create a scaler for inverse-transforming predictions (needed for error calculation)
    all_keys = list(ScalerData_dict.keys())
    scaler_data = ScalerData_dict[f'{all_keys[0]}'].iloc[GLOBAL_SETTINGS["window_size"]:]
    _, scaler_y, _ = scale_dataset_indiv(scaler_data, target_column="GWL")

    inimax = GLOBAL_SETTINGS['inimax']
    opt_sim_members = np.zeros((len(X_opt), inimax))  # Store predictions for each ensemble member

    for ini in range(inimax):
        # Build and train model for each initialization
        model, history = build_cnn_lstm_model(ini, GLOBAL_SETTINGS, X_train, Y_train, X_val, Y_val)

        # Predict on optimization set and inverse-transform
        opt_sim_n = model.predict(X_opt)
        opt_sim = scaler_y.inverse_transform(opt_sim_n)
        opt_sim_members[:, ini] = opt_sim.reshape(-1,)

    # Aggregate ensemble predictions (median)
    opt_sim_median = np.median(opt_sim_members, axis=1)
    sim = np.asarray(opt_sim_median.reshape(-1, 1))
    obs = np.asarray(scaler_y.inverse_transform(Y_opt.reshape(-1, 1)))  # True values

    # Calculate R² between predictions and observations
    r = stats.linregress(sim[:, 0], obs[:, 0])

    err = sim - obs
    RMSE = np.sqrt(np.mean(err ** 2))
    obs_range = np.max(obs) - np.min(obs)
    nRMSE = RMSE / obs_range

    # Geometric mean ensures both metrics contribute meaningfully, mathematically sound
    score = np.sqrt(r.rvalue ** 2 * (1 - nRMSE))
    print("--- Score: ---")
    print(f"R²: {r.rvalue ** 2:.3f}, RMSE: {RMSE:.3f}, nRMSE: {nRMSE:.3f}, Score: {score:.3f}")
    try:
        return score
    except:
        print("Error in score calculation")
        return 0

    #return score  # Return R² as optimization metric
