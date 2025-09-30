
# --- Imports and setup ---
# Set random seeds for reproducibility
from numpy.random import seed
seed(1+347823)
import tensorflow as tf
tf.random.set_seed(1+63493)

# Standard libraries
import numpy as np
import os, glob, shutil
import random
import pandas as pd
import datetime
from scipy import stats
import matplotlib.pyplot as plt
from uncertainties import unumpy
import geopandas as gpd

# Import MinMaxScaler so scaling functions are callable
from sklearn.preprocessing import MinMaxScaler

# --- Utility Functions ---

def clear_directory(directory_path):
    """
    Removes all files and subdirectories within the specified directory.
    Creates the directory if it does not exist.
    """
    if os.path.exists(directory_path):
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f"Cleared all contents of directory: {directory_path}")
    else:
        print(f"Directory does not exist. Creating directory: {directory_path}")
        os.makedirs(directory_path)

def read_and_process_data(file_path, columns_to_keep, target_column="GWL"):
    """
    Reads and processes experimental data for a given well.
    - Moves target_column to first position.
    - Converts 'Date' to datetime and sets as index.
    - Cleans up 'soil_moisture_xy_l10' and 'elevation_msl' columns if present.
    """
    # Read the CSV file with HPC container compatibility
    try:
        # Strategy 1: Standard read
        exp_data = pd.read_csv(file_path, skipinitialspace=True, index_col='Unnamed: 0')
    except (OSError, IOError) as e:
        if "Communication error" in str(e) or "errno 70" in str(e).lower():
            print(f"🔄 HPC I/O issue for {os.path.basename(file_path)}, trying Python engine...")
            try:
                # Strategy 2: Use python engine (slower but more reliable in containers)
                exp_data = pd.read_csv(file_path, skipinitialspace=True, index_col='Unnamed: 0', engine='python')
                print(f"✓ Successfully read {os.path.basename(file_path)} using Python engine")
            except Exception as e2:
                try:
                    # Strategy 3: Read without index_col first, then process
                    exp_data = pd.read_csv(file_path, skipinitialspace=True, engine='python')
                    if 'Unnamed: 0' in exp_data.columns:
                        exp_data = exp_data.drop(columns=['Unnamed: 0'])
                    print(f"✓ Successfully read {os.path.basename(file_path)} using Python engine (no index_col)")
                except Exception as e3:
                    print(f"❌ All CSV reading strategies failed for {file_path}")
                    print(f"❌ Original error: {e}")
                    print(f"❌ Python engine error: {e2}")  
                    print(f"❌ Final attempt error: {e3}")
                    raise e
        else:
            raise e
    exp_data['Date'] = pd.to_datetime(exp_data['Date'])
    exp_data.set_index('Date', inplace=True)
    exp_data = exp_data.loc["1991-01-01":"2024-12-31"]

    # Replace zero values in 'soil_moisture_xy' with NaN
    if 'soil_moisture_xy_l10' in exp_data.columns:
        exp_data.loc[:, 'soil_moisture_xy_l10'] = exp_data['soil_moisture_xy_l10'].replace(0, np.nan)
        # Drop rows with NaN values in the 'soil_moisture_xy' column
        exp_data = exp_data.dropna(subset=['soil_moisture_xy_l10'])
   
    # Convert elevation to float if needed
    if 'elevation_msl' in exp_data.columns and exp_data["elevation_msl"].dtype == "object":
        exp_data["elevation_msl"] = exp_data["elevation_msl"].str.replace(',', '.').astype(float)
       
    # Keep only specified columns, move target to first
    test_df = exp_data[columns_to_keep]
    if target_column in test_df.columns:
        columns_order = [target_column] + [col for col in columns_to_keep if col != target_column]
        test_df = test_df[columns_order]

    return test_df

# --- Scaling Functions ---

## generate global scaler
def generate_scalers(data_orig):
    """
    Generates MinMaxScalers for features and target.

    Parameters:
    - data_orig: pd.DataFrame, original data with 'GWL' column.

    Returns:
    - scaler_x: MinMaxScaler for features.
    - scaler_y: MinMaxScaler for target.
    """
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    scaler_y.fit(pd.DataFrame(data_orig['GWL']))
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_x.fit(data_orig)
    return scaler_x, scaler_y


def scaler_statics_global(input_dir, static_cols, columns_to_keep):
    """
    Gathers static data from all wells and fits a global MinMaxScaler.

    Parameters:
        input_dir (str): Path pattern to CSV files.
        static_cols (list): List of static feature column names.
        read_and_process_data (function): Function to read and clean each file.
        columns_to_keep (list): Columns to keep when processing files.

    Returns:
        scaler_static (MinMaxScaler): Fitted scaler for static features.
        static_df (pd.DataFrame): Combined static feature values for all wells.
    """
    all_static_data = []
    files_found = glob.glob(input_dir)
    print(f"📁 scaler_statics_global: Found {len(files_found)} files with pattern: {input_dir}")
    
    if not files_found:
        raise ValueError(f"No files found with pattern: {input_dir}. Check if the input directory path is correct.")

    for file in files_found:
        try:
            df = read_and_process_data(file, columns_to_keep)
            if df is not None and not df.empty and len(df) > 0:
                static_row = df[static_cols].iloc[0]
                all_static_data.append(static_row)
                print(f"✓ Processed static data from: {os.path.basename(file)}")
            else:
                print(f"⚠️ Skipped empty file: {os.path.basename(file)}")
        except Exception as e:
            print(f"⚠️ Error processing {os.path.basename(file)}: {e}")
            continue

    static_df = pd.DataFrame(all_static_data)
    print(f"📊 Created static DataFrame with {len(static_df)} rows and {len(static_df.columns) if not static_df.empty else 0} columns")
    
    # Safety check: ensure we have data to fit the scaler
    if static_df.empty or len(static_df) == 0:
        raise ValueError(f"No static data available for scaling. All files may have been skipped due to insufficient data. Check if files exist and have enough data for the specified columns: {static_cols}")
    
    scaler_static = MinMaxScaler()
    scaler_static.fit(static_df)

    return scaler_static, static_df

def scale_dataset_indiv(data_orig, target_column="GWL"):
    """
    Scales the target column and features separately, then combines them.
    Returns both scalers and the scaled DataFrame.
    """
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    scaler_x = MinMaxScaler(feature_range=(-1, 1))

    # Scale target
    scaled_y = pd.DataFrame(
        scaler_y.fit_transform(data_orig[[target_column]]),
        index=data_orig.index,
        columns=[target_column]
    )
    
    # Scale features (all columns EXCEPT the target column)
    feature_cols = [col for col in data_orig.columns if col != target_column]
    scaled_x = pd.DataFrame(
        scaler_x.fit_transform(data_orig[feature_cols]),
        index=data_orig.index,
        columns=feature_cols
    )
    
    # Combine: features first, then target
    scaled_df = pd.concat([scaled_x, scaled_y], axis=1)
    return scaler_x, scaler_y, scaled_df

# --- Data Splitting ---

def split_data(data, window_size, test_start_pd_time, test_end_pd_time):
    '''
    Splits data into training, validation, optimization, and test sets.
    - Training: first 80%
    - Validation: next 10%
    - Optimization: last 10% before test period
    - Test: from test_start_pd_time to test_end_pd_time
    Also returns "extended" sets for sequence generation.
    inspired and adapted from Wunsch et al 2022 on github https://github.com/AndreasWunsch/Long-Term-GWL-Simulations
    '''
    dataset = data[(data.index < test_start_pd_time)]  # Remove test data

    # Split by index
    TrainingData = dataset[0:round(0.8 * len(dataset))]
    ValData = dataset[round(0.8 * len(dataset))+1:round(0.9 * len(dataset))]
    ValData_ext = dataset[round(0.8 * len(dataset))+1-window_size:round(0.9 * len(dataset))] #extend data according to dealys/sequence length
    OptData = dataset[round(0.9 * len(dataset))+1:]
    OptData_ext = dataset[round(0.9 * len(dataset))+1-window_size:] #extend data according to dealys/sequence length
    
    TestData = data[(data.index >= test_start_pd_time) & (data.index <= test_end_pd_time)]
    TestData_ext = pd.concat([dataset.iloc[-window_size:], TestData], axis=0) # extend Testdata to be able to fill sequence later                                              

    return TrainingData, ValData, ValData_ext, OptData, OptData_ext, TestData, TestData_ext

# --- Sequence Preparation ---

def to_sequential(data, window_size):
    '''
    Converts data to sequences for time series models.
    modified after Jason Brownlee and machinelearningmastery.com
    inspired and adapted from Wunsch et al 2022 on github https://github.com/AndreasWunsch/Long-Term-GWL-Simulations
    X: shape (samples, window_size, features)
    Y: shape (samples,)
    '''
    X, Y = list(), list()
    # step over the entire history one time step at a time
    for i in range(len(data)):
        # find the end of this pattern
        end_idx = i + window_size
        # check if we are beyond the dataset
        if end_idx >= len(data):
            break
        # seq_y is always the 'GWL' column, seq_x is all other columns
        seq_x = data.iloc[i:end_idx].drop(columns=['GWL'])
        seq_y = data.iloc[end_idx]['GWL']
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)

# --- Main Data Pipeline ---
def process_data_pipeline(input_dir, columns_to_keep, static_cols, GLOBAL_SETTINGS, scaler_static, target_column = "GWL"):
    """
    Main pipeline that processes and scales data for multiple wells for training, validation, and optimization.
    Assumes:
    - Static features are scaled globally using `scaler_static`.
    - Dynamic features and target are scaled per well.

    Parameters:
    - input_dir: str, directory containing the input files.
    - columns_to_keep: list, columns to retain from the dataset.
    - GLOBAL_SETTINGS: dict, settings for window size, test start, and test end.
    - scaler_x, scaler_y: scalers for feature and target normalization.

    Returns:
    - X_train, Y_train: numpy arrays for training data.
    - X_val, Y_val: numpy arrays for validation data.
    - X_opt, Y_opt: numpy arrays for optimization data.
    - ValData_dict, OptData_dict, TestData_dict: dictionaries containing validation, optimization, and test datasets.
    
    """

    # Initialize containers
    X_train = np.empty((0, GLOBAL_SETTINGS["window_size"], len(columns_to_keep) - 1))
    X_val = np.empty((0, GLOBAL_SETTINGS["window_size"], len(columns_to_keep) - 1))
    X_opt = np.empty((0, GLOBAL_SETTINGS["window_size"], len(columns_to_keep) - 1))
   
    Y_train, Y_val, Y_opt= [], [], []
    
    ScalerData_dict, ValData_dict, OptData_dict, TestData_dict= {}, {}, {}, {}

    

    for file in glob.glob(input_dir):
        interim_data = read_and_process_data(file, columns_to_keep)
        well_id = os.path.basename(file).split('_')[0]

        # --- STATIC SCALING ---
        interim_data_static = interim_data[static_cols].iloc[0:1]
        interim_data_static_n = scaler_static.transform(interim_data_static)

        # Apply scaled static values to all time steps
        interim_data_scaled_statics = interim_data.copy()
        for i, col in enumerate(static_cols):
            interim_data_scaled_statics[col] = interim_data_static_n[0, i]

        # --- DYNAMIC + TARGET SCALING ---
        dynamic_and_target_cols = [col for col in interim_data.columns if col not in static_cols]
        interim_data_dynamic = interim_data_scaled_statics[dynamic_and_target_cols]

        scaler_x, scaler_y, interim_data_dynamic_n = scale_dataset_indiv(
            interim_data_dynamic, target_column=target_column
        )

        # Reattach scaled static features
        interim_data_static_n_full = pd.DataFrame(
            np.repeat(interim_data_static_n, len(interim_data), axis=0),
            columns=static_cols,
            index=interim_data.index
        )
        interim_data_n = pd.concat([interim_data_dynamic_n, interim_data_static_n_full], axis=1)

        # Reorder: dynamic + static + target
        dynamic_cols = [col for col in interim_data_dynamic_n.columns if col != "GWL"]
        ordered_cols = ["GWL"] + dynamic_cols + list(static_cols)
        interim_data_n = interim_data_n[ordered_cols]


        # --- SPLIT UN/SCALED DATA ---
        TrainingData, ValData, ValData_ext, OptData, OptData_ext, TestData, TestData_ext = split_data(
            interim_data, GLOBAL_SETTINGS["window_size"],
            GLOBAL_SETTINGS["test_start"], GLOBAL_SETTINGS["test_end"]
        )

        TrainingData_n, ValData_n, ValData_ext_n, OptData_n, OptData_ext_n, TestData_n, TestData_ext_n = split_data(
            interim_data_n, GLOBAL_SETTINGS["window_size"],
            GLOBAL_SETTINGS["test_start"], GLOBAL_SETTINGS["test_end"]
        )

         # Convert data to sequential format
        X_train_interim, Y_train_interim = to_sequential(TrainingData_n, GLOBAL_SETTINGS["window_size"])
        X_val_interim, Y_val_interim = to_sequential(ValData_n, GLOBAL_SETTINGS["window_size"])
        X_opt_interim, Y_opt_interim = to_sequential(OptData_n, GLOBAL_SETTINGS["window_size"])
        X_test_interim, Y_test_interim = to_sequential(TestData_n, GLOBAL_SETTINGS["window_size"])

        # Append to training arrays (with dimension safety checks)
        if X_train_interim.size > 0 and X_train_interim.ndim == 3:
            X_train = np.concatenate((X_train, X_train_interim), axis=0)
            Y_train = np.concatenate((Y_train, Y_train_interim), axis=0)
        else:
            print(f"⚠️  Skipping {well_id} train data - insufficient data for window_size={GLOBAL_SETTINGS['window_size']}")
            
        if X_val_interim.size > 0 and X_val_interim.ndim == 3:
            X_val = np.concatenate((X_val, X_val_interim), axis=0)
            Y_val = np.concatenate((Y_val, Y_val_interim), axis=0)
        else:
            print(f"⚠️  Skipping {well_id} val data - insufficient data for window_size={GLOBAL_SETTINGS['window_size']}")
            
        if X_opt_interim.size > 0 and X_opt_interim.ndim == 3:
            X_opt = np.concatenate((X_opt, X_opt_interim), axis=0)
            Y_opt = np.concatenate((Y_opt, Y_opt_interim), axis=0)
        else:
            print(f"⚠️  Skipping {well_id} opt data - insufficient data for window_size={GLOBAL_SETTINGS['window_size']}")

        # Save to dictionaries

        ScalerData_dict[f'all_Dataframe_{well_id}'] = interim_data
    
        ValData_dict[f'obs_Dataframe_{well_id}'] = ValData
        ValData_dict[f'X_val_{well_id}'] = X_val_interim
        ValData_dict[f'Y_val_{well_id}'] = Y_val_interim

        OptData_dict[f'obs_Dataframe_{well_id}'] = OptData
        OptData_dict[f'X_opt_{well_id}'] = X_opt_interim
        OptData_dict[f'Y_opt_{well_id}'] = Y_opt_interim

        TestData_dict[f'obs_Dataframe_{well_id}'] = TestData
        TestData_dict[f'X_test_{well_id}'] = X_test_interim
        TestData_dict[f'Y_test_{well_id}'] = Y_test_interim

    # Final safety check - ensure we have enough data for training
    if X_train.shape[0] == 0:
        raise ValueError(f"No training data available after processing. Window size {GLOBAL_SETTINGS['window_size']} might be too large for the available data.")
    if X_val.shape[0] == 0:
        raise ValueError(f"No validation data available after processing. Window size {GLOBAL_SETTINGS['window_size']} might be too large for the available data.")
        
    print(f"✓ Data pipeline completed: Train={X_train.shape}, Val={X_val.shape}, Opt={X_opt.shape}")

    return X_train, Y_train, X_val, Y_val, X_opt, Y_opt, ScalerData_dict, ValData_dict, OptData_dict, TestData_dict

# def process_data_pipeline(input_dir, columns_to_keep, GLOBAL_SETTINGS):
#     """
#     Process data from multiple files into sequential datasets for training, validation, and optimization.

#     Parameters:
#     - input_dir: str, directory containing the input files.
#     - columns_to_keep: list, columns to retain from the dataset.
#     - GLOBAL_SETTINGS: dict, settings for window size, test start, and test end.
#     - scaler_x, scaler_y: scalers for feature and target normalization.

#     Returns:
#     - X_train, Y_train: numpy arrays for training data.
#     - X_val, Y_val: numpy arrays for validation data.
#     - X_opt, Y_opt: numpy arrays for optimization data.
#     - ValData_dict, OptData_dict, TestData_dict: dictionaries containing validation, optimization, and test datasets.
#     """

#     # Initialize empty arrays and dictionaries
#     X_train = np.empty((0, GLOBAL_SETTINGS["window_size"], len(columns_to_keep) - 1))
#     X_val = np.empty((0, GLOBAL_SETTINGS["window_size"], len(columns_to_keep) - 1))
#     X_opt = np.empty((0, GLOBAL_SETTINGS["window_size"], len(columns_to_keep) - 1))
#     Y_train, Y_val, Y_opt = [], [], []
#     ScalerData_dict, ValData_dict, OptData_dict, TestData_dict = {}, {}, {}, {}

#     for file in glob.glob(input_dir):
#         # Read and preprocess data for this well
#         interim_data = read_and_process_data(file, columns_to_keep)
#         well_id = os.path.basename(file).split('_')[0]

#         # Scale data for this well
#         scaler_x, scaler_y, interim_data_n = scale_dataset_indiv(interim_data, target_column="GWL")

#         # Split data into training, validation, optimization, and test sets
#         TrainingData, ValData, ValData_ext, OptData, OptData_ext, TestData, TestData_ext = split_data(
#             interim_data, 
#             GLOBAL_SETTINGS["window_size"], 
#             GLOBAL_SETTINGS["test_start"], 
#             GLOBAL_SETTINGS["test_end"]
#         )
#         TrainingData_n, ValData_n, ValData_ext_n, OptData_n, OptData_ext_n, TestData_n, TestData_ext_n = split_data(
#             interim_data_n, 
#             GLOBAL_SETTINGS["window_size"], 
#             GLOBAL_SETTINGS["test_start"], 
#             GLOBAL_SETTINGS["test_end"]
#         )

#         # Convert to sequences for model input
#         X_train_interim, Y_train_interim = to_sequential(TrainingData_n, GLOBAL_SETTINGS["window_size"])
#         X_val_interim, Y_val_interim = to_sequential(ValData_n, GLOBAL_SETTINGS["window_size"])
#         X_opt_interim, Y_opt_interim = to_sequential(OptData_n, GLOBAL_SETTINGS["window_size"])
#         X_test_interim, Y_test_interim = to_sequential(TestData_n, GLOBAL_SETTINGS["window_size"])

#         # Add to global arrays
#         X_train = np.concatenate((X_train, X_train_interim), axis=0)
#         Y_train = np.concatenate((Y_train, Y_train_interim), axis=0)
#         X_val = np.concatenate((X_val, X_val_interim), axis=0)
#         Y_val = np.concatenate((Y_val, Y_val_interim), axis=0)
#         X_opt = np.concatenate((X_opt, X_opt_interim), axis=0)
#         Y_opt = np.concatenate((Y_opt, Y_opt_interim), axis=0)

#         # Store per-well data for later analysis or inverse-scaling
#         ScalerData_dict[f'all_Dataframe_{well_id}'] = interim_data
#         ValData_dict[f'obs_Dataframe_{well_id}'] = ValData
#         ValData_dict[f'X_val_{well_id}'] = X_val_interim
#         ValData_dict[f'Y_val_{well_id}'] = Y_val_interim

#         OptData_dict[f'obs_Dataframe_{well_id}'] = OptData
#         OptData_dict[f'X_opt_{well_id}'] = X_opt_interim
#         OptData_dict[f'Y_opt_{well_id}'] = Y_opt_interim

#         TestData_dict[f'obs_Dataframe_{well_id}'] = TestData
#         TestData_dict[f'X_test_{well_id}'] = X_test_interim
#         TestData_dict[f'Y_test_{well_id}'] = Y_test_interim

#     # Returns: arrays for model training, plus dicts for per-well access
#     return X_train, Y_train, X_val, Y_val, X_opt, Y_opt, ScalerData_dict, ValData_dict, OptData_dict, TestData_dict
