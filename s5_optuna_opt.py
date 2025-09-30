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
import pickle
import tempfile

# Keras imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import load_model

# Optuna for hyperparameter optimization
import optuna
# from optuna.integration import TensorFlowPruningHook  # Not needed for basic optimization
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler

# Scikit-learn for data prep
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import shap

# List available GPUs (for info/debug)
gpus = tf.config.experimental.list_physical_devices('GPU')

# Import project modules
from s1_data_preparation import *
from s2_model_utils import *

# Configuration constants (not window-size dependent)
COLUMNS_TO_KEEP = [
    'tas_3x3_mean',
    'pr_3x3_mean',
    'hurs_3x3_mean',
    'soil_mois_composite_3x3_mean_0-30',
    'soil_mois_composite_3x3_mean_0-60',
    'soil_mois_composite_3x3_mean_0-90',
    'elevation_msl',
    'MW_muGOK',
    'distance_to_waterwork_km',
    'kf_remapped',
    'GWL'
]

STATIC_COLS = [
    'elevation_msl',
    'MW_muGOK',
    'distance_to_waterwork_km',
    'kf_remapped',
]

INPUT_DIR = "data_filtered_anthro/*.csv"
SELECTED_FILES_LIST = None  # Will be set by the main script to filter files

def get_filtered_input_dir():
    """
    Get input directory or file list - handles both scenarios
    """
    if SELECTED_FILES_LIST is not None:
        return SELECTED_FILES_LIST
    else:
        return INPUT_DIR

def objective(trial):
    """
    Optuna objective function for hyperparameter optimization
    Only optimizes: windowsize, densesize, batchsize, filters
    """
    # Suggest only the 4 hyperparameters to optimize
    densesize_int = trial.suggest_int('densesize', 8, 256, step=8)
    windowsize_int = trial.suggest_int('windowsize', 30, 80, step=2)
    #batchsize_int = trial.suggest_categorical('batchsize', [16, 32, 64, 128])
    batchsize_int = trial.suggest_categorical('batchsize', [16, 32, 64, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576]
)
    filters_int = trial.suggest_int('filters', 16, 256, step=8)
    
    print(f"\nTrial {trial.number}:")
    print(f"densesize: {densesize_int}, windowsize: {windowsize_int}")
    print(f"batchsize: {batchsize_int}, filters: {filters_int}")
    
    try:
        # Set model/global settings for this trial
        GLOBAL_SETTINGS = {
            'inimax': 2,  # Reduced for faster optimization, increase for final runs
            'batch_size': batchsize_int,
            'kernel_size': 3,
            'dense_size': densesize_int,
            'filters': filters_int,
            'window_size': windowsize_int,
            'clip_norm': True,
            'clip_value': 1,
            'epochs': 150,  # Reduced for faster trials, increase for final runs
            'learning_rate': 1e-5,  # Fixed learning rate from original script
            'test_start': pd.to_datetime('2018-01-01', format='%Y-%m-%d'),
            'test_end': pd.to_datetime('2024-09-01', format='%Y-%m-%d'),
            'num_cnn_layers': 5,
            'lstm_units': [32, 16],  # Fixed LSTM units from original script
            'trial': trial  # Pass trial for pruning
        }
        
        # Get input directory for this trial
        filtered_input = get_filtered_input_dir()
        print(f"✓ Trial {trial.number}: Starting with params - densesize={densesize_int}, windowsize={windowsize_int}, batchsize={batchsize_int}, filters={filters_int}")
        
        # Check file count
        if isinstance(filtered_input, str):
            actual_files = glob.glob(filtered_input)
            print(f"✓ Trial {trial.number}: Found {len(actual_files)} files with pattern {filtered_input}")
            if len(actual_files) < 50:
                print(f"⚠️ Trial {trial.number}: WARNING - Only {len(actual_files)} files found, expected ~217")
        
        # Prepare scaler for static features (needs to be done per trial since windowsize affects it)
        print(f"✓ Trial {trial.number}: Preparing static scaler...")
        
        # Handle both file list and glob pattern cases
        if isinstance(filtered_input, list):
            # If we have a list of files, create a temporary glob pattern
            # by joining them with a pattern that scaler_statics_global can use
            temp_dir = tempfile.mkdtemp()
            temp_pattern = os.path.join(temp_dir, "*.csv")
            
            # Copy files to temp directory for scaler_statics_global
            for file_path in filtered_input:
                shutil.copy2(file_path, temp_dir)
            
            scaler_static, _ = scaler_statics_global(
                input_dir=temp_pattern,
                static_cols=STATIC_COLS,
                columns_to_keep=COLUMNS_TO_KEEP
            )
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
        else:
            # Standard glob pattern case
            scaler_static, _ = scaler_statics_global(
                input_dir=filtered_input,
                static_cols=STATIC_COLS,
                columns_to_keep=COLUMNS_TO_KEEP
            )
        
        print(f"✓ Trial {trial.number}: Static scaler prepared successfully")
        
        # Prepare data for this trial (windowsize affects preprocessing)
        print(f"✓ Trial {trial.number}: Processing data pipeline...")
        
        # Handle both file list and glob pattern cases for data processing
        if isinstance(filtered_input, list):
            # If we have a list of files, create a temporary glob pattern
            temp_dir = tempfile.mkdtemp()
            temp_pattern = os.path.join(temp_dir, "*.csv")
            
            # Copy files to temp directory for process_data_pipeline
            for file_path in filtered_input:
                shutil.copy2(file_path, temp_dir)
            
            X_train, Y_train, X_val, Y_val, X_opt, Y_opt, ScalerData_dict, ValData_dict, OptData_dict, TestData_dict = process_data_pipeline(
                input_dir=temp_pattern,
                columns_to_keep=COLUMNS_TO_KEEP,
                static_cols=STATIC_COLS,
                GLOBAL_SETTINGS=GLOBAL_SETTINGS,
                scaler_static=scaler_static,
                target_column="GWL"
            )
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
        else:
            # Standard glob pattern case
            X_train, Y_train, X_val, Y_val, X_opt, Y_opt, ScalerData_dict, ValData_dict, OptData_dict, TestData_dict = process_data_pipeline(
                input_dir=filtered_input,
                columns_to_keep=COLUMNS_TO_KEEP,
                static_cols=STATIC_COLS,
                GLOBAL_SETTINGS=GLOBAL_SETTINGS,
                scaler_static=scaler_static,
                target_column="GWL"
            )
        
        print(f"✓ Trial {trial.number}: Data pipeline completed - X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
        
        # Create scaler for inverse-transforming predictions
        all_keys = list(ScalerData_dict.keys())
        scaler_data = ScalerData_dict[f'{all_keys[0]}'].iloc[GLOBAL_SETTINGS["window_size"]:]
        _, scaler_y, _ = scale_dataset_indiv(scaler_data, target_column="GWL")
        
        inimax = GLOBAL_SETTINGS['inimax']
        opt_sim_members = np.zeros((len(X_opt), inimax))
        
        # Train ensemble with early stopping based on validation performance
        for ini in range(inimax):
            print(f"Training ensemble member {ini+1}/{inimax}")
            
            # Build and train model using existing function from s2_model_utils
            model, history = build_cnn_lstm_model(
                ini, GLOBAL_SETTINGS, X_train, Y_train, X_val, Y_val
            )
            
            # Check if trial was pruned during training
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            # Predict on optimization set
            opt_sim_n = model.predict(X_opt)
            opt_sim = scaler_y.inverse_transform(opt_sim_n)
            opt_sim_members[:, ini] = opt_sim.reshape(-1,)
            
            # Early pruning check after each ensemble member
            if ini == 0:  # After first member, do intermediate evaluation
                temp_sim = np.asarray(opt_sim.reshape(-1, 1))
                temp_obs = np.asarray(scaler_y.inverse_transform(Y_opt.reshape(-1, 1)))
                
                # Calculate R² and NSE for intermediate evaluation
                temp_r = stats.linregress(temp_sim[:, 0], temp_obs[:, 0])
                temp_r2 = temp_r.rvalue ** 2
                
                # Calculate NSE
                temp_obs_mean = np.mean(temp_obs)
                temp_numerator = np.sum((temp_obs - temp_sim) ** 2)
                temp_denominator = np.sum((temp_obs - temp_obs_mean) ** 2)
                temp_nse = 1 - (temp_numerator / temp_denominator)
                
                # Early stopping based on R² threshold
                if temp_r2 < 0.09:
                    print(f"🚫 Trial {trial.number} stopped early: R² = {temp_r2:.3f} < 0.09 threshold after ensemble member {ini+1}")
                    trial.set_user_attr('early_stop_reason', f'R2_too_low_{temp_r2:.3f}')
                    trial.set_user_attr('stopped_at_ensemble_member', ini+1)
                    raise optuna.exceptions.TrialPruned()
                
                # Intermediate score: geometric mean of R² and NSE
                # Ensure both metrics are positive for geometric mean calculation
                temp_r2_clipped = max(temp_r2, 0.001)  # Avoid zero/negative values
                temp_nse_clipped = max(temp_nse, 0.001)  # Avoid zero/negative values
                temp_score = np.sqrt(temp_r2_clipped * temp_nse_clipped)
                
                # Report intermediate value for pruning
                trial.report(temp_score, ini)
                
                # Check if trial should be pruned by Optuna's pruner
                if trial.should_prune():
                    print(f"🚫 Trial {trial.number} pruned by Optuna after ensemble member {ini+1} (Score: {temp_score:.3f})")
                    trial.set_user_attr('early_stop_reason', 'optuna_pruner')
                    trial.set_user_attr('stopped_at_ensemble_member', ini+1)
                    raise optuna.exceptions.TrialPruned()
        
        # Aggregate ensemble predictions (median)
        opt_sim_median = np.median(opt_sim_members, axis=1)
        sim = np.asarray(opt_sim_median.reshape(-1, 1))
        obs = np.asarray(scaler_y.inverse_transform(Y_opt.reshape(-1, 1)))
        
        # Calculate R² and NSE (Nash-Sutcliffe Efficiency)
        r = stats.linregress(sim[:, 0], obs[:, 0])
        r2 = r.rvalue ** 2
        
        # Calculate NSE (Nash-Sutcliffe Efficiency)
        # NSE = 1 - (sum of squared residuals) / (sum of squared deviations from mean)
        obs_mean = np.mean(obs)
        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - obs_mean) ** 2)
        nse = 1 - (numerator / denominator)
        
        # Target: geometric mean of R² and NSE
        # Ensure both metrics are positive for geometric mean calculation
        r2_clipped = max(r2, 0.001)  # Avoid zero/negative values
        nse_clipped = max(nse, 0.001)  # Avoid zero/negative values
        score = np.sqrt(r2_clipped * nse_clipped)
        
        # Calculate additional metrics for reporting
        err = sim - obs
        RMSE = np.sqrt(np.mean(err ** 2))
        obs_range = np.max(obs) - np.min(obs)
        nRMSE = RMSE / obs_range
        
        # Store individual metrics as user attributes for logging
        trial.set_user_attr('r2', float(r2))
        trial.set_user_attr('nse', float(nse))
        trial.set_user_attr('rmse', float(RMSE))
        trial.set_user_attr('nrmse', float(nRMSE))
        
        print(f"Trial {trial.number} results:")
        print(f"R²: {r2:.3f}, NSE: {nse:.3f}, Score (√(R²×NSE)): {score:.3f}")
        print(f"RMSE: {RMSE:.3f}, nRMSE: {nRMSE:.3f}")
        
        return score
        
    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial.number} was pruned")
        raise
    except Exception as e:
        print(f"🚨 CRITICAL ERROR in trial {trial.number}: {str(e)}")
        print(f"🚨 Error type: {type(e).__name__}")
        print(f"🚨 Trial parameters: densesize={densesize_int}, windowsize={windowsize_int}, batchsize={batchsize_int}, filters={filters_int}")
        import traceback
        print("🚨 Full traceback:")
        traceback.print_exc()
        
        # Log to file for later analysis
        error_log_file = f"trial_{trial.number}_error.log"
        with open(error_log_file, 'w') as f:
            f.write(f"Trial {trial.number} Error Report\n")
            f.write(f"Parameters: densesize={densesize_int}, windowsize={windowsize_int}, batchsize={batchsize_int}, filters={filters_int}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Error type: {type(e).__name__}\n")
            f.write("Full traceback:\n")
            traceback.print_exc(file=f)
        
        print(f"🚨 Error details saved to: {error_log_file}")
        return 0.0  # Return poor score for failed trials

# Removed custom model building functions - using existing build_cnn_lstm_model from s2_model_utils

def run_optimization():
    """
    Run the Optuna optimization with pruning
    """
    print("Starting Optuna optimization...")
    
    # Create study with pruning
    pruner = MedianPruner(
        n_startup_trials=25,  # Number of trials before pruning starts (25 random exploration)
        n_warmup_steps=5,     # Number of steps before pruning evaluation
        interval_steps=1      # Interval between pruning evaluations
    )
    
    # Alternative: SuccessiveHalvingPruner for more aggressive pruning
    # pruner = SuccessiveHalvingPruner()
    
    sampler = TPESampler(seed=42)  # Tree-structured Parzen Estimator
    
    study = optuna.create_study(
        direction='maximize',  # We want to maximize our score
        pruner=pruner,
        sampler=sampler,
        study_name='cnn_lstm_optimization'
    )
    
    # Save study progress
    study_file = f'optuna_study_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    
    try:
        # Run optimization
        print("Starting Optuna optimization...")
        study.optimize(
            objective,
            n_trials=150,  # Maximum trials with early stopping via pruning
            timeout=None,   # Set timeout in seconds if needed
            callbacks=[lambda study, trial: save_study_progress(study, study_file)]
        )
        
        # Print results
        print("\nOptimization completed!")
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        
        print("\nBest trial:")
        trial = study.best_trial
        print(f"Score: {trial.value:.4f}")
        print("Best params:")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")
        
        # Save final study
        with open(f'final_{study_file}', 'wb') as f:
            pickle.dump(study, f)
        
        # Create visualization if optuna-dashboard is available
        try:
            import optuna.visualization as vis
            
            # Create plots
            fig1 = vis.plot_optimization_history(study)
            fig1.write_html('optimization_history.html')
            
            fig2 = vis.plot_param_importances(study)
            fig2.write_html('param_importances.html')
            
            fig3 = vis.plot_parallel_coordinate(study)
            fig3.write_html('parallel_coordinate.html')
            
            print("\nVisualization files created:")
            print("- optimization_history.html")
            print("- param_importances.html")
            print("- parallel_coordinate.html")
            
        except ImportError:
            print("Install plotly for visualizations: pip install plotly")
        
        return study
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        print(f"Saving current progress to {study_file}")
        save_study_progress(study, study_file)
        return study

def save_study_progress(study, filename):
    """
    Save study progress periodically
    """
    with open(filename, 'wb') as f:
        pickle.dump(study, f)

def load_study(filename):
    """
    Load a saved study
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Run optimization
    study = run_optimization()
    
    print("\nOptimization finished. Use the best parameters for your final model training.")
