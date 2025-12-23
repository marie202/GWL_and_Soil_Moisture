# CNN model for groundwater level prediction
# Uses all files for training in the range 1990 to 2016
# Data from https://github.com/AndreasWunsch/Long-Term-GWL-Simulations

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
input_dir = "data_217/*.csv"

# List of well IDs from filenames
well_ids = [os.path.basename(file).split('_')[0] for file in glob.glob(input_dir)]

# Count files
files = glob.glob("data_217/*.csv")  
print("Total number of files: ", len(files))

# --- Model hyperparameters (from previous optimization) ---
best_params =  {'batchsize': 494.6087670307434, 'densesize': 191.78224186685904, 'filters': 72.51321797642332, 'windowsize': 58} #'target': 0.374491149402591, 'params': {
# new hyperparams
# best_params =  {'batchsize': 32, 'densesize': 120, 'filters': 96, 'windowsize': 46} #"value": 0.6131111482852543,

# --- Load optimized hyperparameters from optimization script ---
# Set the directory where the optimization results are saved
# opt_model_dir = "model_runs/BayesOpt/BayesOpt_CNN_LSTM_5layer_11params_bayes_opt"

# # Load best parameters (expects s2_model_utils.py to provide load_best_params)
# best_params = load_best_params(opt_model_dir)

# if best_params is None:
#     raise RuntimeError(f"Could not load best hyperparameters from {opt_model_dir}")

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
    'clip_norm': True,
    'clip_value': 1,
    'epochs': 250,
    'learning_rate': 1e-5,
    'test_start': pd.to_datetime('2019-01-01', format='%Y-%m-%d'),
    'test_end': pd.to_datetime('2024-12-31', format='%Y-%m-%d'),
    'num_cnn_layers': 6,
    'lstm_units': [32, 16], #[32, 16],
    'use_cnn_only': True,  # Set to True to use CNN-only model (without LSTM), False to use CNN-LSTM
    'model_dir_note': "AllFeat_SM_217files_test_period_start_2019_ini10_CNN_only_6L_58W"
}



# Features to use for training (last entry must be 'GWL')
columns_to_keep = [
    'tas_3x3_mean',
    'pr_3x3_sum',
    #'pr_3x3_sum_logit',
    'hurs_3x3_mean',
    #'soil_moisture_xy_general',
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
# Always append a timestamp to avoid accidental overwrites if note isn't changed
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Use appropriate model name based on architecture
model_type = "CNN" if GLOBAL_SETTINGS.get('use_cnn_only', False) else "CNN_LSTM"
model_dir = os.path.join(
    path,
    f"BB_{model_type}_{GLOBAL_SETTINGS['num_cnn_layers']}layer_{len(columns_to_keep)}params_{GLOBAL_SETTINGS['model_dir_note']}_{timestamp}"
)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
print("Model directory: ", model_dir)

# --- Training data preparation ---

# # '''
# ## Zufallsdatein zum cnn durchlauf auswählen
# def clear_directory(directory_path):
#     """
#     Removes all files and subdirectories within the specified directory.
#     """
#     if os.path.exists(directory_path):
#         for item in os.listdir(directory_path):
#             item_path = os.path.join(directory_path, item)
#             if os.path.isfile(item_path):
#                 os.remove(item_path)
#             elif os.path.isdir(item_path):
#                 shutil.rmtree(item_path)
#         print(f"Cleared all contents of directory: {directory_path}")
#     else:
#         print(f"Directory does not exist. Creating directory: {directory_path}")
#         os.makedirs(directory_path)

# directory = "data/"
# #well_id_number = training_files[number].split('/')[2].split('_')[0]

# sample_directory = os.path.join(directory, "working_samples")#+str(well_id_number))

# if not os.path.exists(sample_directory):
#     os.makedirs(sample_directory)


# clear_directory(sample_directory)
# use_files = 20


# # Alle Dateien im Verzeichnis abrufen
# all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
# # Prüfen, ob es mindestens 60 Dateien gibt
# num_files = min(use_files, len(all_files))  # Falls weniger als 60 Dateien existieren
# # Zufällig 60 Dateien auswählen
# random_files = random.sample(all_files, num_files)

# # Ausgabe der ausgewählten Dateien
# print(f"Ausgewählte {num_files} Dateien:")
#     # Dateien kopieren
# for file in random_files:
#     shutil.copy(os.path.join(directory, file), os.path.join(sample_directory, file))

# print(f"{num_files} zufällige Dateien wurden nach '{sample_directory}' kopiert.")



scaler_static, _ = scaler_statics_global(
    input_dir=input_dir,
    static_cols=static_cols,
    columns_to_keep=columns_to_keep
)

##############################
# Read all training files into one DataFrame

# Training data directory
source_dir = "data_217"
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

#model_path = model_dir+"/"

# Load the best model (median_idx) from training history
# import ast
# training_history_path = os.path.join(model_path, 'traininghistory_CNN_BB.txt')
# with open(training_history_path, 'r') as f:
#     lines = f.readlines()
#     for i, line in enumerate(lines):
#         if 'median_idx' in line:
#             median_idx = eval(lines[i + 1])
#             break
# Load model weights for median run (for reference)
# model = load_model(model_dir + f'/model_weights_BB_run_{median_idx}.keras')

# Run test set simulation and plot results, get scores for all wells
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

# Convert scores dictionary to DataFrame
df = pd.concat(scores_dict, names=["ID"]).reset_index(level=1, drop=True).reset_index()

# Display result summary (avoiding full dataframe print on HPC)
try:
    print(df)
except Exception as e:
    print(f"Summary: {len(df)} wells analyzed")
    print(f"R² range: {df['R2'].min():.3f} to {df['R2'].max():.3f}")
    print(f"RMSE range: {df['RMSE'].min():.3f} to {df['RMSE'].max():.3f}")

# Save scores DataFrame to file
path = model_dir+"/"
file = "scores_df.csv"
try:
    df.to_csv(os.path.join(path, file), index=False)
except Exception as e:
    print(f"[WARNING] pandas to_csv failed ({e}). Falling back to manual CSV writer.")
    csv_path = os.path.join(path, file)
    try:
        with open(csv_path, 'w') as f:
            f.write(','.join(map(str, df.columns)) + '\n')
            for row in df.itertuples(index=False):
                f.write(','.join(map(str, row)) + '\n')
        print(f"[INFO] Wrote CSV via fallback to {csv_path}")
    except Exception as e2:
        print(f"[ERROR] Fallback CSV write failed: {e2}")

df = pd.read_csv(os.path.join(model_dir, "scores_df.csv"))
# Print R² and RMSE statistics
print("R² Statistics:")
print("Mean:", df['R2'].mean())
print("Min:", df['R2'].min())
print("Max:", df['R2'].max())

print("RMSE Statistics:")
print("Mean:", df['RMSE'].mean())
print("Min:", df['RMSE'].min())
print("Max:", df['RMSE'].max())

# Load scores DataFrame for plotting
df = pd.read_csv(os.path.join(model_dir, "scores_df.csv"))

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



median_model = load_model(model_path+f'model_weights_BB_run_{median_idx}.keras')

# Concatenate all test data for feature importance
X_test_all = np.concatenate([TestData_dict[f'X_test_{well_id}'] for well_id in well_ids], axis=0)
Y_test_all = np.concatenate([TestData_dict[f'Y_test_{well_id}'] for well_id in well_ids], axis=0)


# Compute and save feature importance
feature_importance(
    median_model,
    X_test_all,
    Y_test_all,
    scaler_y,
    columns_to_keep,
    model_path,         # output_dir
    "feature_importance.txt"         # output_filename
)

# --- SHAP Values ---




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

# --- SHAP Values by Landcover Class ---
print("\n--- Computing SHAP values grouped by landcover class ---")
try:
    metadata_path = "BB_GW_wells_metadata_coords_4_visual_anthropo_217.csv"
    metadata_df = pd.read_csv(metadata_path)
    well_to_landcover = {}
    for _, row in metadata_df.iterrows():
        well_id = str(row['ID'])
        landcover = str(row['dominant_land_cover_2018'])
        if pd.notna(landcover) and landcover != 'nan':
            well_to_landcover[well_id] = landcover
    landcover_groups = {}
    for well_id in well_ids:
        landcover = well_to_landcover.get(well_id, 'unknown')
        if landcover not in landcover_groups:
            landcover_groups[landcover] = []
        landcover_groups[landcover].append(well_id)
    sorted_landcovers = sorted(landcover_groups.keys())
    print(f"Found {len(sorted_landcovers)} landcover classes:")
    for lc in sorted_landcovers:
        print(f"  {lc}: {len(landcover_groups[lc])} wells")
    all_shap_by_landcover = {}
    all_input_by_landcover = {}
    for landcover in sorted_landcovers:
        group_well_ids = landcover_groups[landcover]
        print(f"\nComputing SHAP values for landcover '{landcover}' ({len(group_well_ids)} wells)...")
        X_test_landcover = np.concatenate(
            [TestData_dict[f'X_test_{well_id}'] for well_id in group_well_ids if f'X_test_{well_id}' in TestData_dict],
            axis=0
        )
        if X_test_landcover.shape[0] == 0:
            print(f"  WARNING: No test data found for landcover '{landcover}', skipping...")
            continue
        shap_vals_lc, X_test_last_lc = compute_and_save_shap_values_robust(
            median_model=median_model,
            X_train=X_train,
            X_test_all=X_test_landcover,
            model_dir=model_path,
            columns_to_keep=columns_to_keep,
            nsamples=100,
            background_size=100,
            use_kmeans_background=True,
            hide_logging=True,
            stability_check=True
        )
        all_shap_by_landcover[landcover] = shap_vals_lc
        all_input_by_landcover[landcover] = X_test_last_lc
        print(f"  Completed: {shap_vals_lc.shape[0]} samples")
    output_path = os.path.join(model_dir, "shapvalues_by_landcover.txt")
    print(f"\nSaving all SHAP values by landcover to: {output_path}")
    with open(output_path, "w") as f:
        f.write("# SHAP values grouped by landcover class\n")
        f.write(f"# Total landcover classes: {len(sorted_landcovers)}\n")
        f.write(f"# Format: Each landcover section has 'landcover_<class>', 'shap_vals', 'input_data' sections\n\n")
        for landcover in sorted_landcovers:
            if landcover not in all_shap_by_landcover:
                continue
            shap_vals_lc = all_shap_by_landcover[landcover]
            input_data_lc = all_input_by_landcover[landcover]
            f.write(f'landcover_{landcover}\n')
            f.write(f'n_samples_{landcover}: {shap_vals_lc.shape[0]}\n')
            f.write('shap_vals\n')
            for row in shap_vals_lc:
                f.write(' '.join(map(str, row)) + '\n')
            f.write('input_data\n')
            for row in input_data_lc:
                f.write(' '.join(map(str, row)) + '\n')
            f.write('\n')
    print(f"Successfully saved SHAP values by landcover to: {output_path}")
    print(f"Total samples across all landcover classes: {sum(v.shape[0] for v in all_shap_by_landcover.values())}")
    print("\n--- Plotting SHAP values by landcover ---")
    try:
        feature_cols = [col for col in columns_to_keep if "GWL" not in col]
        custom_labels_lc = []
        for col in feature_cols:
            if col == 'tas_3x3_mean':
                custom_labels_lc.append('T')
            elif col == 'pr_3x3_sum':
                custom_labels_lc.append('P')
            elif col == 'pr_3x3_sum_logit':
                custom_labels_lc.append('P')
            elif col == 'hurs_3x3_mean':
                custom_labels_lc.append('rH')
            elif col == 'soil_mois_composite_3x3_mean_0-30':
                custom_labels_lc.append('SM 0-30cm')
            elif col == 'soil_mois_composite_3x3_mean_0-60':
                custom_labels_lc.append('SM 0-60cm')
            elif col == 'soil_mois_composite_3x3_mean_0-90':
                custom_labels_lc.append('SM 0-90cm')
            elif col == 'elevation_msl':
                custom_labels_lc.append('Elev')
            elif col == 'MW_muGOK':
                custom_labels_lc.append('GWT')
            elif col == 'distance_to_waterwork_km':
                custom_labels_lc.append('DistWW')
            elif col == 'kf_remap_number':
                custom_labels_lc.append('kf')
            elif col.endswith('_percent'):
                custom_labels_lc.append(col.replace('_percent', ''))
            else:
                custom_labels_lc.append(col)
        plot_shap_by_landcover(
            txt_path=output_path,
            columns_to_keep=feature_cols,
            model_dir=model_dir,
            custom_labels=custom_labels_lc,
            show_plot=True,
        )
    except Exception as e:
        print(f"[WARNING] Plotting error in plot_shap_by_landcover: {e}")
except Exception as e:
    print(f"[WARNING] Error computing SHAP values by landcover: {e}")


# --- If there is a direct SHAP plotting block, wrap it as well ---
# Example:
# try:
#     shap.summary_plot(shap_vals, x, feature_names=columns_to_keep, show=False)
#     plt.title(f"SHAP Values for well BB{str(well_ids[number_of_well-1])}")
#     plt.xlabel("SHAP value (impact on GWL)")
#     plt.savefig('./model_runs/BB_CNN_opt/shap_values.png', dpi=300)
#     plt.show()
# except Exception as e:
#     print(f"[WARNING] Plotting error in direct SHAP plot: {e}")