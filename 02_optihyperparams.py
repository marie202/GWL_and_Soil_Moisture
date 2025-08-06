# HYPERPARAM OPTIMIZATION
#> github https://github.com/AndreasWunsch/Long-Term-GWL-Simulations

## First, lets import all neeeded libraries

# --- Imports (keep only what is needed here) ---
import os
import glob
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import shap
from scipy import stats

# Set seeds for reproducibility
tf.random.set_seed(1 + 63493)
np.random.seed(1 + 347823)
random.seed(1 + 347823)

from s1_data_preparation import *
from s2_model_utils import *
from s3_plotting_functions import *
from s4_bayesian_opt import *

# --- Configuration ---
input_dir = "data/working_samples/*.csv"

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

# Read metadata from Stations/wells
## List stations

# List well IDs
well_ids = [os.path.basename(file).split('_')[0] for file in glob.glob(input_dir)]
print(f"Total number of wells: {len(well_ids)}")

# # Global settings (default, will be updated after optimization)
GLOBAL_SETTINGS = {
    'inimax': 5,  # number of initializations for uncertainty
    'kernel_size': 3,  # must be odd
    'clip_norm': True,
    'clip_value': 1,
    'epochs': 250,
    'learning_rate': 1e-5,
    'test_start': pd.to_datetime('2018-01-01', format='%Y-%m-%d'),
    'test_end': pd.to_datetime('2024-09-01', format='%Y-%m-%d'),
    'num_cnn_layers': 5,
    'lstm_units': [32, 16], #[32, 16],
    'model_dir_note': "bayes_opt"
}


# --- Prepare model directory ---
path = "model_runs/BayesOpt"
model_dir = os.path.join(
    path,   
    f"BayesOpt_CNN_LSTM_{GLOBAL_SETTINGS['num_cnn_layers']}layer_{len(columns_to_keep)}params_{GLOBAL_SETTINGS['model_dir_note']}"
)
os.makedirs(model_dir, exist_ok=True)


# # --- Data Loading ---
# X_train, Y_train, X_val, Y_val, X_opt, Y_opt, ScalerData_dict, ValData_dict, OptData_dict, TestData_dict = process_data_pipeline(
#     input_dir=input_dir,
#     columns_to_keep=columns_to_keep,
#     GLOBAL_SETTINGS=GLOBAL_SETTINGS)

# --- Hyperparameter Optimization ---

# Define Parameter bounds:
# Bounded region of parameter space
bounds = {
        'batchsize': (16, 512), #batchsize_int, #16-128 
        'densesize': (16, 256), #densesize_int, 
        'filters': (16, 256), #filters_int, 
        'windowsize': (20, 70), # for sake of simplicity set it to 20 --> later test (1, 52), #seqlength_int,
    }
    
    
####################################
## -- run bayesian optimization --
####################################



## define acq. function
## expected improvement (probably the most common acquisition function) 
## xi=0.05  #  Prefer exploitation (xi=0.0) / Prefer exploration (xi=0.1)

acquisition_function = acquisition.ExpectedImprovement(xi=0.0)

optimizer = BayesianOptimization(
            f= bayesOpt_function, #optimized function
            pbounds=bounds, #parameter bounds
            acquisition_function=acquisition_function,
            random_state=1, 
            verbose = 0 # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent, verbose = 2 prints everything
            )


optimizer.maximize(
                init_points=50, #20 #steps of random exploration (random starting points before bayesopt(?))
                n_iter=5, # steps of bayesian optimization

)

# Evaluate Results of Hyperparameter tuning
for i, res in enumerate(optimizer.res):
   print("Iteration {}: \n\t{}".format(i, res))

print(optimizer.max)



#get best values from optimizer
batchsize_int = int(optimizer.max.get("params").get("batchsize"))
densesize_int = int(optimizer.max.get("params").get("densesize"))
windowsize_int = int(optimizer.max.get("params").get("windowsize"))
filters_int = int(optimizer.max.get("params").get("filters"))

print(" --- Best values: ---")
print(f"batchsize_int: {batchsize_int}, densesize_int: {densesize_int}, filters_int: {filters_int}, windowsize_int: {windowsize_int}")

# --- Save optimization results ---
import json
import pickle
from datetime import datetime

# Create results dictionary
optimization_results = {
    'best_params': {
        'batchsize': batchsize_int,
        'densesize': densesize_int,
        'windowsize': windowsize_int,
        'filters': filters_int
    },
    'best_score': optimizer.max.get("target"),
    'optimization_history': optimizer.res,
    'bounds': bounds,
    'global_settings': GLOBAL_SETTINGS,
    'columns_to_keep': columns_to_keep,
    'timestamp': datetime.now().isoformat(),
    'model_dir': model_dir
}

# Save as JSON (human readable)
json_file_path = os.path.join(model_dir, 'optimization_results.json')
with open(json_file_path, 'w') as f:
    # Convert datetime objects to strings for JSON serialization
    json.dump(optimization_results, f, indent=2, default=str)

# Save as pickle (preserves all object types)
pickle_file_path = os.path.join(model_dir, 'optimization_results.pkl')
with open(pickle_file_path, 'wb') as f:
    pickle.dump(optimization_results, f)

# Save optimizer object itself (for resuming optimization if needed)
optimizer_file_path = os.path.join(model_dir, 'optimizer.pkl')
with open(optimizer_file_path, 'wb') as f:
    pickle.dump(optimizer, f)

print(f"\n--- Optimization results saved ---")
print(f"JSON file: {json_file_path}")
print(f"Pickle file: {pickle_file_path}")
print(f"Optimizer file: {optimizer_file_path}")

# Also save best parameters in a simple format for easy loading
best_params_simple = {
    'batchsize': batchsize_int,
    'densesize': densesize_int,
    'windowsize': windowsize_int,
    'filters': filters_int,
    'best_score': optimizer.max.get("target")
}

simple_params_file = os.path.join(model_dir, 'best_params.json')
with open(simple_params_file, 'w') as f:
    json.dump(best_params_simple, f, indent=2)

print(f"Simple params file: {simple_params_file}")



