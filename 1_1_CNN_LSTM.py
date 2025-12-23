# CNN-LSTM hybrid model for groundwater level modelling


# --- Imports and setup ---
from numpy.random import seed
seed(1+347823)  # Set numpy random seed for reproducibility
import tensorflow as tf
tf.random.set_seed(1+63493)  # Set tensorflow random seed

import matplotlib
matplotlib.use('Agg') # make sure to use Agg backend for plotting -> so it runs on HPC
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib as mpl
mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.sans-serif'] = [
    'Liberation Sans', 'Arial', 'Helvetica', 'Nimbus Sans', 'FreeSans', 'DejaVu Sans'
]
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Standard libraries
import numpy as np
import os, glob, shutil, re
import random
import pandas as pd
import datetime
from scipy import stats
import matplotlib.pyplot as plt
# Removed uncertainties package - not used in this script
# Removed geopandas - not used in this script

# Keras/tensorflow layers
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import load_model

# Bayesian optimization
# from bayes_opt import BayesianOptimization, acquisition

# Scikit-learn for data prep
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import shap  # For SHAP values

# List available GPUs (for info)
# Configure TensorFlow to use GPU memory growth (prevents allocating all GPU memory at once)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)

if gpus:
    try:
        # Enable memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Configured {len(gpus)} GPU(s) with memory growth enabled")
        
        # Print GPU details
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"    Details: {gpu_details}")
            except:
                pass
    except RuntimeError as e:
        print(f"Error configuring GPUs: {e}")
else:
    print("WARNING: No GPUs detected. Training will run on CPU.")
    print("This may be due to:")
    print("  1. Container not run with --nv flag")
    print("  2. CUDA libraries not properly mounted")
    print("  3. TensorFlow not built with GPU support")
    print("  4. CUDA version mismatch")

# Import project modules
from s1_data_preparation import *
from s3_plotting_functions import *
from s2_model_utils import *


# Set working directory
pwd = os.getcwd()
print("Working directory set to:", pwd)

# --- Data and settings ---
# Read metadata from Stations/wells
# List of all wells

# Input data directory (all CSVs)
input_dir = "data/*.csv"

# List of well IDs from filenames
well_ids = [os.path.basename(file).split('_')[0] for file in glob.glob(input_dir)]

# Count files
files = glob.glob("data/*.csv")  
print("Total number of files: ", len(files))

# --- Model hyperparameters (from previous optimization) ---
best_params =  {'batchsize': 494, 'densesize': 192, 'filters': 192, 'windowsize': 58} 


# Extract and ensure integer type for model usage
batchsize_int = int(best_params['batchsize'])
densesize_int = int(best_params['densesize'])
windowsize_int = int(best_params['windowsize'])
filters_int = int(best_params['filters'])

#print(f"Loaded optimized hyperparameters from {opt_model_dir}:")
print(f"  batchsize: {batchsize_int}")
print(f"  densesize: {densesize_int}")
print(f"  windowsize: {windowsize_int}")
print(f"  filters: {filters_int}")


# Global settings for model building
GLOBAL_SETTINGS = {
    'inimax': 10,  # number of initializations for uncertainty
    'batch_size': batchsize_int,
    'kernel_size': 3,  # must be odd
    'dense_size': densesize_int,
    'filters': filters_int,
    'window_size': windowsize_int,
    'clip_norm': True, # False
    'clip_value': 1,
    'epochs': 250,
    'learning_rate': 1e-5,
    'test_start': pd.to_datetime('2019-01-01', format='%Y-%m-%d'),
    'test_end': pd.to_datetime('2024-12-31', format='%Y-%m-%d'),
    'num_cnn_layers': 6,
    'lstm_units': [128, 64], #[32, 16],
    'model_dir_note':"AllFeat_SM_ini2_217files_10ini" #AllFeat_SM_217files_test_period_start_2019_ini10_6layer_new_hyperparams_test"
}


# Features to use for training (last entry must be 'GWL')
columns_to_keep = [
    'tas_3x3_mean',
    'pr_3x3_mean',
    'hurs_3x3_mean',
    'soil_mois_composite_3x3_mean_0-30',
    'soil_mois_composite_3x3_mean_0-60',
    'soil_mois_composite_3x3_mean_0-90',
    'elevation_msl',
    'MW_muGOK',
    'distance_to_waterwork_km',
    'kf_remap_number',
    'GWL'
]
print("Columns to train on: ", columns_to_keep)


static_cols = [
    'elevation_msl',
    'MW_muGOK',
    'distance_to_waterwork_km',
    'kf_remap_number',
    ] 


# --- Model output directory ---
path = "model_runs/"
model_dir = os.path.join(path, f"BB_CNN_LSTM_{GLOBAL_SETTINGS['num_cnn_layers']}layer_{len(columns_to_keep)}params_{GLOBAL_SETTINGS['model_dir_note']}")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
print("Model directory: ", model_dir)


scaler_static, _ = scaler_statics_global(
    input_dir=input_dir,
    static_cols=static_cols,
    columns_to_keep=columns_to_keep
)

##############################
# Read all training files into one DataFrame

# Training data directory
source_dir = "data_filtered_anthro"
training_files = glob.glob(source_dir+"/*.csv")
target_column = "GWL"

# Extract well IDs from filenames in working_samples
well_ids = []
print(os.getcwd())
for file_name in os.listdir(source_dir):
    if os.path.isfile(os.path.join(source_dir, file_name)):
        match = re.match(r"^(\d+)_weeklyData_", file_name)
        if match:
            well_ids.append(match.group(1))

# Initialize arrays for training, validation, optimization (scaled and unscaled)
X_train_n = np.empty((0, GLOBAL_SETTINGS["window_size"], len(columns_to_keep)-1))  
Y_train_n = [] 
X_val_n = np.empty((0, GLOBAL_SETTINGS["window_size"], len(columns_to_keep)-1))  
Y_val_n  = [] 
X_opt_n = np.empty((0, GLOBAL_SETTINGS["window_size"], len(columns_to_keep)-1))  
Y_opt_n  = [] 
X_val_modperf= np.empty((0, GLOBAL_SETTINGS["window_size"], len(columns_to_keep)-1))  
X_opt_modperf= np.empty((0, GLOBAL_SETTINGS["window_size"], len(columns_to_keep)-1))  
Y_val_modperf, Y_opt_modperf  = [],[] 

# Dictionaries to store per-well data
ValData_dict= {}
OptData_dict= {}
TestData_dict= {}

# Loop over all input files and prepare data
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
    ordered_cols = dynamic_cols + list(static_cols) + ["GWL"]
    interim_data_n = interim_data_n[ordered_cols]



    # Split data into train/val/opt/test (unscaled)
    TrainingData, ValData, ValData_ext, OptData, OptData_ext, TestData, TestData_ext = split_data(
        interim_data, 
        GLOBAL_SETTINGS["window_size"],
        GLOBAL_SETTINGS["test_start"],
        GLOBAL_SETTINGS["test_end"]
    )          

    # Split data into train/val/opt/test (scaled)
    TrainingData_n, ValData_n, ValData_ext_n, OptData_n, OptData_ext_n, TestData_n, TestData_ext_n = split_data(
        interim_data_n, 
        GLOBAL_SETTINGS["window_size"],
        GLOBAL_SETTINGS["test_start"],
        GLOBAL_SETTINGS["test_end"]
    )          

    # Convert to sequential format for model input
    X_train_interim_n, Y_train_interim_n = to_sequential(TrainingData_n, GLOBAL_SETTINGS["window_size"],)
    X_val_interim_n, Y_val_interim_n = to_sequential(ValData_n, GLOBAL_SETTINGS["window_size"],)
    X_opt_interim_n, Y_opt_interim_n = to_sequential(OptData_n, GLOBAL_SETTINGS["window_size"],)  
    X_test_interim_n, Y_test_interim_n = to_sequential(TestData_n, GLOBAL_SETTINGS["window_size"],)

    # Unscaled data for evaluation
    X_val_interim, Y_val_interim = to_sequential(ValData, GLOBAL_SETTINGS["window_size"],)
    X_opt_interim, Y_opt_interim = to_sequential(OptData, GLOBAL_SETTINGS["window_size"],)  

    # Append unscaled validation/optimization data
    X_val_modperf = np.concatenate((X_val_modperf, X_val_interim), axis=0)
    Y_val_modperf = np.concatenate((Y_val_modperf, Y_val_interim), axis=0)
    X_opt_modperf = np.concatenate((X_opt_modperf, X_opt_interim), axis=0)
    Y_opt_modperf = np.concatenate((Y_opt_modperf, Y_opt_interim), axis=0)

    # Append scaled data for training
    X_train_n = np.concatenate((X_train_n, X_train_interim_n), axis=0)
    Y_train_n = np.concatenate((Y_train_n, Y_train_interim_n), axis=0)

    X_val_n = np.concatenate((X_val_n, X_val_interim_n), axis=0)
    Y_val_n = np.concatenate((Y_val_n, Y_val_interim_n), axis=0)

    X_opt_n = np.concatenate((X_opt_n, X_opt_interim_n), axis=0)
    Y_opt_n = np.concatenate((Y_opt_n, Y_opt_interim_n), axis=0)

    # Store per-well data in dictionaries
    ValData_dict[f'obs_Dataframe_{well_id}'] = ValData
    ValData_dict[f'X_val_{well_id}'] = X_val_interim_n
    ValData_dict[f'Y_val_{well_id}'] = Y_val_interim_n

    OptData_dict[f'obs_Dataframe_{well_id}'] = OptData
    OptData_dict[f'X_opt_{well_id}'] = X_opt_interim_n
    OptData_dict[f'Y_opt_{well_id}'] = Y_opt_interim_n

    TestData_dict[f'obs_Dataframe_{well_id}'] = TestData
    TestData_dict[f'X_test_{well_id}'] = X_test_interim_n
    TestData_dict[f'Y_test_{well_id}'] = Y_test_interim_n

# --- Use function pipeline for data preparation  ---
X_train, Y_train, X_val, Y_val, X_opt, Y_opt, ScalerData_dict, ValData_dict, OptData_dict, TestData_dict = process_data_pipeline(
    input_dir=input_dir,
    columns_to_keep=columns_to_keep,
    static_cols=static_cols,
    GLOBAL_SETTINGS=GLOBAL_SETTINGS,
    scaler_static=scaler_static,
    target_column = "GWL"
)

# Print number of training, validation, test, and optimization samples before training
try:
    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of validation samples: {X_val.shape[0]}")
    print(f"Number of test samples: {TestData_dict[f'X_test_{well_ids[0]}'].shape[0] if 'X_test_' + well_ids[0] in TestData_dict else 'N/A'}")
    print(f"Number of optimization samples: {X_opt.shape[0]}")
except Exception as e:
    print(f"[WARNING] Error printing dataset sizes: {e}")


# Save test data and scaler for later use (e.g. plotting)
import joblib
testdata_dict_path = os.path.join(model_dir, "TestData_dict.pkl")
joblib.dump(TestData_dict, testdata_dict_path)
scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")
joblib.dump(scaler_y, scaler_y_path)

# --- Model training and evaluation ---

inimax = GLOBAL_SETTINGS['inimax']
scaler_data = ScalerData_dict[f'all_Dataframe_{well_ids[0]}'].iloc[GLOBAL_SETTINGS["window_size"]:]

# Train model and get results
median_idx, scores, sim1, obs1, inimax, sim_members, sim_members_uncertainty, sim_mean_uncertainty, final_loss, final_val_loss, val_loss_final_epoch = simulate_testset(
    path=path, 
    model_dir=model_dir, 
    X_train=X_train, 
    Y_train=Y_train, 
    X_val=X_val, 
    Y_val=Y_val,
    scaler_data=scaler_data, 
    inimax=inimax, 
    BL_abbr="BB",
    GLOBAL_SETTINGS=GLOBAL_SETTINGS,
)

# Plot training and validation loss curves
try:
    plot_loss_curves(model_dir, BL_abbr = "BB", show_plot=True)
except Exception as e:
    print(f"[WARNING] Plotting error in plot_loss_curves: {e}")

# --- Plotting and evaluation functions ---

# Run test set simulation and plot results, get scores for all wells
try:
    scores_dict = simulate_plot_wells_testset_conf_int(
        well_ids=well_ids,
        TestData_dict=TestData_dict,
        GLOBAL_SETTINGS=GLOBAL_SETTINGS,
        model_path = model_dir+"/",
        # scaler_y=scaler_y,
        number_of_wells=len(well_ids),
        inimax=inimax,
        columns_to_keep=columns_to_keep,
        save_fig=True
    )
except Exception as e:
    print(f"[ERROR] Failed to run test set simulation and plotting: {e}")
    print("[WARNING] Continuing with remaining analysis...")
    scores_dict = {}  # Set empty dict to avoid downstream errors


# Convert scores dictionary to DataFrame
try:
    df = pd.concat(scores_dict, names=["ID"]).reset_index(level=1, drop=True).reset_index()
except Exception as e:
    print(f"[ERROR] Failed to convert scores dictionary to DataFrame: {e}")
    print("[WARNING] Skipping score analysis...")
    df = pd.DataFrame()  # Empty DataFrame to avoid downstream errors

# Display result summary (avoiding full dataframe print on HPC)
try:
    if not df.empty:
        # Use to_string() with explicit formatting to avoid pandas internal module issues
        print(df.to_string())
    else:
        print("Summary: 0 wells analyzed (DataFrame is empty)")
except Exception as e:
    # Fallback: try to print basic info without using pandas formatting
    try:
        if not df.empty:
            print(f"Summary: {len(df)} wells analyzed")
            print(f"R² range: {df['R2'].min():.3f} to {df['R2'].max():.3f}")
            print(f"RMSE range: {df['RMSE'].min():.3f} to {df['RMSE'].max():.3f}")
        else:
            print("Summary: 0 wells analyzed (DataFrame is empty)")
    except Exception as e2:
        print(f"[WARNING] Could not display DataFrame summary: {e2}")
        print(f"DataFrame shape: {df.shape if hasattr(df, 'shape') else 'unknown'}")


# Save scores DataFrame to file
path = model_dir+"/"
file = "scores_df.csv"
try:
    if not df.empty:
        df.to_csv(os.path.join(path, file), index=False)
        # Print R² and RMSE statistics
        print("R² Statistics:")
        print("Mean:", df['R2'].mean())
        print("Min:", df['R2'].min())
        print("Max:", df['R2'].max())

        print("RMSE Statistics:")
        print("Mean:", df['RMSE'].mean())
        print("Min:", df['RMSE'].min())
        print("Max:", df['RMSE'].max())
    else:
        print("[WARNING] Scores DataFrame is empty, skipping statistics and CSV save")
except Exception as e:
    print(f"[WARNING] Failed to save scores or print statistics: {e}")
    try:
        # Try to load existing file if it exists
        if os.path.exists(os.path.join(model_dir, "scores_df.csv")):
            df = pd.read_csv(os.path.join(model_dir, "scores_df.csv"))
            print("Loaded existing scores_df.csv file")
        else:
            print("[WARNING] No scores file available")
    except Exception as e2:
        print(f"[ERROR] Could not load scores file: {e2}")

# Load scores DataFrame for plotting (if available)
try:
    if os.path.exists(os.path.join(model_dir, "scores_df.csv")):
        df = pd.read_csv(os.path.join(model_dir, "scores_df.csv"))
    elif df.empty:
        print("[WARNING] No scores data available for plotting")
except Exception as e:
    print(f"[WARNING] Could not load scores for plotting: {e}")

try:
    # Plot R2 and RMSE boxplots
    plot_r2_rmse_boxplots(
        df,
        model_dir, 
        fontsize=16, 
        show_plot=True
    )

    plot_r2_rmse_nse_bias_boxplot(
        df,
        model_dir, 
        fontsize=16, 
        show_plot=True
    )
    pass
except Exception as e:
    print(f"An error occurred while plotting R² and RMSE boxplots: {e}")

# --- Evaluation: Feature Importance and SHAP values ---

# --- Feature Importance ---

model_path = model_dir+"/"

# Apply BatchNormalization axis fix patch before loading the median model
# This fixes the issue where axis is saved as a list [2] but Keras 3.x expects an integer
try:
    from keras.src.layers.normalization.batch_normalization import BatchNormalization
    from keras.src.saving import serialization_lib
    
    # Patch BatchNormalization.from_config to handle axis conversion
    if not hasattr(BatchNormalization, '_patched_for_axis_fix'):
        original_bn_from_config = BatchNormalization.from_config
        
        @classmethod
        def patched_bn_from_config(cls, config):
            """Patched from_config to convert axis from list to integer"""
            if isinstance(config, dict) and 'axis' in config:
                if isinstance(config['axis'], list):
                    # Convert list to integer (take first element, or -1 if empty)
                    config = config.copy()
                    config['axis'] = config['axis'][0] if config['axis'] else -1
            return original_bn_from_config(config)
        
        BatchNormalization.from_config = patched_bn_from_config
        BatchNormalization._patched_for_axis_fix = True
        
        # Also patch the deserialize function to fix axis before it reaches from_config
        original_deserialize = serialization_lib.deserialize_keras_object
        
        def fix_initializer_config(init_config):
            """Fix initializer configs that are saved as dicts"""
            if isinstance(init_config, dict):
                # If it's a dict with module/class_name structure, deserialize it
                if 'module' in init_config and 'class_name' in init_config:
                    try:
                        from keras.src.saving import serialization_lib
                        return serialization_lib.deserialize_keras_object(init_config)
                    except:
                        # If deserialization fails, try to extract just the class name
                        class_name = init_config.get('class_name', '')
                        inner_config = init_config.get('config', {})
                        # Return a simplified version that Keras can handle
                        if class_name:
                            try:
                                from keras import initializers
                                init_class = getattr(initializers, class_name, None)
                                if init_class:
                                    return init_class(**inner_config) if inner_config else init_class()
                            except:
                                pass
            return init_config
        
        def fix_keras2_to_keras3_compat(config_dict, max_depth=50, current_depth=0, visited=None):
            """Recursively fix Keras 2.x -> 3.x compatibility issues with recursion depth limit"""
            # Prevent infinite recursion
            if current_depth > max_depth:
                print(f"[WARNING] Maximum recursion depth ({max_depth}) reached in fix_keras2_to_keras3_compat, returning config as-is")
                return config_dict
            
            if not isinstance(config_dict, dict):
                return config_dict
            
            # Track visited objects to prevent circular references
            if visited is None:
                visited = set()
            config_id = id(config_dict)
            if config_id in visited:
                # Circular reference detected, return as-is
                return config_dict
            visited.add(config_id)
            
            try:
                config_dict = config_dict.copy()
                class_name = config_dict.get('class_name', '')
                inner_config = config_dict.get('config', {})
                
                if class_name == 'BatchNormalization' and isinstance(inner_config, dict):
                    # Fix BatchNormalization axis: convert list to integer
                    if 'axis' in inner_config and isinstance(inner_config['axis'], list):
                        inner_config = inner_config.copy()
                        inner_config['axis'] = inner_config['axis'][0] if inner_config['axis'] else -1
                        config_dict['config'] = inner_config
                
                # Fix initializer configs in layer configs
                if isinstance(inner_config, dict):
                    initializer_keys = ['kernel_initializer', 'bias_initializer', 'recurrent_initializer', 
                                       'beta_initializer', 'gamma_initializer', 'moving_mean_initializer', 
                                       'moving_variance_initializer']
                    for key in initializer_keys:
                        if key in inner_config:
                            inner_config[key] = fix_initializer_config(inner_config[key])
                    config_dict['config'] = inner_config
                
                # Recursively fix nested configs
                if 'layers' in config_dict and isinstance(config_dict['layers'], list):
                    config_dict['layers'] = [fix_keras2_to_keras3_compat(layer, max_depth, current_depth+1, visited) for layer in config_dict['layers']]
                elif 'config' in config_dict and isinstance(config_dict['config'], dict):
                    config_dict['config'] = fix_keras2_to_keras3_compat(config_dict['config'], max_depth, current_depth+1, visited)
                
                return config_dict
            finally:
                # Remove from visited set when done (allows same object to be processed again at different depth)
                visited.discard(config_id)
        
        def patched_deserialize(config, custom_objects=None, safe_mode=False, **kwargs):
            """Patched deserialize to fix Keras 2.x -> 3.x compatibility issues"""
            if isinstance(config, dict):
                config = fix_keras2_to_keras3_compat(config)
            return original_deserialize(config, custom_objects=custom_objects, safe_mode=safe_mode, **kwargs)
        
        serialization_lib.deserialize_keras_object = patched_deserialize
        print("✓ Applied BatchNormalization axis fix patch before loading median model")
except Exception as e:
    print(f"[WARNING] Could not apply BatchNormalization patch: {e}")

# Load median model and perform feature importance/SHAP analysis
try:
    median_model = load_model(model_path+f'model_weights_BB_run_{median_idx}.keras')

    # Concatenate all test data for feature importance
    X_test_all = np.concatenate([TestData_dict[f'X_test_{well_id}'] for well_id in well_ids], axis=0)
    Y_test_all = np.concatenate([TestData_dict[f'Y_test_{well_id}'] for well_id in well_ids], axis=0)

    # Compute and save feature importance
    try:
        feature_importance(
            median_model,
            X_test_all,
            Y_test_all,
            scaler_y,
            columns_to_keep,
            model_path,         # output_dir
            "feature_importance.txt"         # output_filename
        )
    except Exception as e:
        print(f"[WARNING] Failed to compute feature importance: {e}")

    # --- SHAP Values ---
    try:
        shap_vals, X_test_last = compute_and_save_shap_values_robust(
            median_model=median_model,
            X_train=X_train,
            X_test_all=X_test_all,
            model_dir=model_path,
            columns_to_keep=columns_to_keep,
            nsamples=100,  # Number of samples for SHAP computation
            background_size=100,  # Size of background dataset (k-means centers)
            use_kmeans_background=True,  # Use k-means for better background selection
            hide_logging=True,
            stability_check=True  # Enable automatic clipping of extreme values
        )
    except Exception as e:
        print(f"[WARNING] Failed to compute SHAP values: {e}")
        shap_vals = None
        X_test_last = None
except Exception as e:
    print(f"[ERROR] Failed to load median model or perform analysis: {e}")
    print("[WARNING] Skipping feature importance and SHAP analysis...")
    median_model = None
    shap_vals = None
    X_test_last = None

# Plot SHAP values if they were computed successfully
if shap_vals is not None:
    try:
        # Create custom labels dynamically based on columns_to_keep (excluding GWL)
        feature_cols = [col for col in columns_to_keep if "GWL" not in col]
        custom_labels = []
        
        for col in feature_cols:
            if col == 'tas_3x3_mean':
                custom_labels.append('T')
            elif col == 'pr_3x3_sum':
                custom_labels.append('P')
            elif col == 'pr_3x3_sum_logit':
                custom_labels.append('P')
            elif col == 'hurs_3x3_mean':
                custom_labels.append('rH')
            elif col == 'soil_mois_composite_3x3_mean_0-30':
                custom_labels.append('SM 0-30cm')
            elif col == 'soil_mois_composite_3x3_mean_0-60':
                custom_labels.append('SM 0-60cm')
            elif col == 'soil_mois_composite_3x3_mean_0-90':
                custom_labels.append('SM 0-90cm')
            elif col == 'elevation_msl':
                custom_labels.append('Elev')
            elif col == 'MW_muGOK':
                custom_labels.append('GWT')
            elif col == 'distance_to_waterwork_km':
                custom_labels.append('DistWW')
            elif col == 'kf_remap_number':
                custom_labels.append('kf')
            elif col.endswith('_percent'):
                custom_labels.append(col.replace('_percent', ''))
            else:
                custom_labels.append(col)

        plot_shap_from_txt(
            txt_path=os.path.join(model_dir, f"shapvalues.txt"),
            columns_to_keep=feature_cols,
            model_dir=model_dir,
            custom_labels=custom_labels,
            show_plot=True,
        )
    except Exception as e:
        print(f"[WARNING] Plotting error in plot_shap_from_txt: {e}")
else:
    print("[WARNING] SHAP values not available, skipping SHAP plotting")
