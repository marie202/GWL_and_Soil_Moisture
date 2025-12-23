# HYPERPARAM OPTIMIZATION WITH OPTUNA
# Based on the original Bayesian optimization script but using Optuna with pruning

## First, lets import all needed libraries

# --- Imports (keep only what is needed here) ---
import os
import glob
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
import json
import pickle
import datetime
from pathlib import Path

# Optuna imports
import optuna
# from optuna.integration import TFKerasPruningCallback  # Not needed for basic optimization
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Set seeds for reproducibility
tf.random.set_seed(1 + 63493)
np.random.seed(1 + 347823)
random.seed(1 + 347823)

# from s1_data_preparation import *
# from s2_model_utils import *
# from s3_plotting_functions import *
from s5_optuna_opt import objective  # Import the objective function

# --- Configuration ---
# Anchor all relative paths to this script's directory so output isn't affected
# by the current working directory (e.g., SLURM job scripts).
BASE_DIR = Path(__file__).resolve().parent

input_dir = str(BASE_DIR / "data_217/*.csv")

# Use ALL files for optimization (better quality, more robust results)
all_files = glob.glob(input_dir)
print(f"Using ALL {len(all_files)} files for optimization")
print("This will take longer but gives more robust hyperparameter selection")

# Update s5_optuna_opt.py to use all files
import s5_optuna_opt
s5_optuna_opt.INPUT_DIR = input_dir
s5_optuna_opt.SELECTED_FILES_LIST = None  # Use all files, no filtering

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
    'kf_remap_number',  # Note: changed from 'kf_remap_number' to match your s4 script
    'GWL'
]
print("Columns to train on: ", columns_to_keep)

# Update columns in s5_optuna_opt.py
s5_optuna_opt.COLUMNS_TO_KEEP = columns_to_keep

static_cols = [
    'elevation_msl',
    'MW_muGOK',
    'distance_to_waterwork_km',
    'kf_remap_number',  # Note: changed from 'kf_remap_number' to match your s4 script
] 

# Update static cols in s5_optuna_opt.py
s5_optuna_opt.STATIC_COLS = static_cols

# List well IDs from all files
well_ids = [os.path.basename(f).split('_')[0] for f in all_files]
print(f"Total number of wells for optimization: {len(well_ids)}")

# Global settings (similar to original, will be used as base for optimization)
GLOBAL_SETTINGS = {
    'inimax': 3,  # Reduced for optimization (increase for final runs)
    'kernel_size': 3,  # must be odd
    'clip_norm': True,
    'clip_value': 1,
    'epochs': 250,  # Reduced for optimization
    'learning_rate': 1e-5,
    'test_start': pd.to_datetime('2018-01-01', format='%Y-%m-%d'),
    'test_end': pd.to_datetime('2024-09-01', format='%Y-%m-%d'),
    'num_cnn_layers': 6,
    'lstm_units': [64, 32],
    'model_dir_note': f"optuna_allfiles_{len(all_files)}wells_v1_6layers_64_32"
}

# --- Prepare timestamped model directory ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
path = BASE_DIR / "model_runs" / "OptunaOpt"
model_dir = path / f"OptunaOpt_{timestamp}_CNN_LSTM_{GLOBAL_SETTINGS['num_cnn_layers']}layer_{len(columns_to_keep)}params_{GLOBAL_SETTINGS['model_dir_note']}"
os.makedirs(model_dir, exist_ok=True)
print(f"✓ Results will be saved to: {model_dir}")
print(f"✓ Timestamp: {timestamp}")

def save_iteration_result(study, trial):
    """Save each iteration result - console output only"""
    # Console output only
    if trial.value is not None:
        duration_min = trial.duration.total_seconds() / 60 if trial.duration else 0
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        output = f"[{current_time}] Trial {trial.number}: Score={trial.value:.6f}"
        
        # Add individual metrics if available
        if hasattr(trial, 'user_attrs') and trial.user_attrs:
            metrics_str = []
            if 'r2' in trial.user_attrs:
                metrics_str.append(f"R²={trial.user_attrs['r2']:.3f}")
            if 'nse' in trial.user_attrs:
                metrics_str.append(f"NSE={trial.user_attrs['nse']:.3f}")
            if metrics_str:
                output += f" ({', '.join(metrics_str)})"
        
        output += f", Duration={duration_min:.1f}min, State={trial.state.name}"
        print(output)
    else:
        duration_min = trial.duration.total_seconds() / 60 if trial.duration else 0
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] Trial {trial.number}: Duration={duration_min:.1f}min, {trial.state.name}")

def save_study_progress(study):
    """Save current study progress including best trial info and all trials"""
    print(f"🔍 save_study_progress called - Study has {len(study.trials)} trials")
    
    # Collect all trials with detailed info
    all_trials = []
    for trial in study.trials:
        trial_info = {
            'trial_number': trial.number,
            'value': trial.value if trial.value is not None else None,
            'params': trial.params,
            'state': trial.state.name,
            'duration_seconds': trial.duration.total_seconds() if trial.duration else None,
            'is_best': trial.number == study.best_trial.number if study.best_trial else False
        }
        
        # Add individual metrics if available
        if hasattr(trial, 'user_attrs') and trial.user_attrs:
            metrics = {}
            if 'r2' in trial.user_attrs:
                metrics['r2'] = trial.user_attrs['r2']
            if 'nse' in trial.user_attrs:
                metrics['nse'] = trial.user_attrs['nse']
            if 'rmse' in trial.user_attrs:
                metrics['rmse'] = trial.user_attrs['rmse']
            if 'nrmse' in trial.user_attrs:
                metrics['nrmse'] = trial.user_attrs['nrmse']
            if metrics:
                trial_info['metrics'] = metrics
            
            # Add early stopping information
            if 'early_stop_reason' in trial.user_attrs:
                trial_info['early_stop_reason'] = trial.user_attrs['early_stop_reason']
            if 'stopped_at_ensemble_member' in trial.user_attrs:
                trial_info['stopped_at_ensemble_member'] = trial.user_attrs['stopped_at_ensemble_member']
        
        all_trials.append(trial_info)
    
    # Sort completed trials by value (best first)
    completed_trials = [t for t in all_trials if t['value'] is not None]
    completed_trials.sort(key=lambda x: x['value'], reverse=True)
    failed_trials = [t for t in all_trials if t['value'] is None]
    
    # Only save comprehensive trials log (single file with all trial info)
    all_trials_file = model_dir / 'all_trials_log.json'
    print(f"📝 Writing {len(completed_trials)} completed + {len(failed_trials)} failed trials to: {all_trials_file}")
    
    with open(all_trials_file, 'w') as f:
        json.dump({
            'last_updated': datetime.datetime.now().isoformat(),
            'summary': {
                'total_trials': len(study.trials),
                'completed_trials': len(completed_trials),
                'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
            },
            'best_trial': {
                'number': study.best_trial.number if study.best_trial else None,
                'value': study.best_value if study.best_trial else None,
                'params': study.best_params if study.best_trial else None,
                'marked_as': '🏆 BEST TRIAL 🏆'
            } if study.best_trial else None,
            'all_trials_ranked': completed_trials + failed_trials
        }, f, indent=2)
    
    print(f"✅ JSON file written successfully to: {all_trials_file}")
    print(f"[Optuna] Study progress: {len(completed_trials)} complete, {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])} pruned, {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])} failed")
    if study.best_trial:
        print(f"[Optuna] Current best: Trial {study.best_trial.number} with score {study.best_value:.6f}")

def create_study():
    """Create new in-memory study"""
    print("[Optuna] Creating new study")
    
    # Create study with pruning (in-memory, no database)
    pruner = MedianPruner(
        n_startup_trials=25,  # Number of trials before pruning starts (25 random exploration)
        n_warmup_steps=5,     # Number of steps before pruning evaluation
        interval_steps=1      # Interval between pruning evaluations
    )
    
    sampler = TPESampler(seed=42)  # Tree-structured Parzen Estimator
    
    study = optuna.create_study(
        direction='maximize',  # We want to maximize our score
        pruner=pruner,
        sampler=sampler
    )
    
    return study

# --- Hyperparameter Optimization with Optuna ---

print("[Optuna] Starting hyperparameter optimization...")

# Create study
study = create_study()

# Define callback to save results after each trial
def trial_callback(study, trial):
    print(f"🔄 Callback triggered for trial {trial.number}")
    save_iteration_result(study, trial)
    
    # Save study progress after EVERY trial for real-time monitoring
    try:
        save_study_progress(study)
        print(f"💾 Trial {trial.number} results saved to log file")
    except Exception as e:
        print(f"❌ Error saving trial {trial.number}: {e}")
        import traceback
        traceback.print_exc()

# Run optimization
n_trials = 150  # Maximum trials with early stopping via pruning (25 random + up to 125 TPE-guided)
timeout = None  # Set timeout in seconds if needed (e.g., 3600 for 1 hour)

print(f"[Optuna] Running optimization for {n_trials} trials...")
print(f"[Optuna] Results will be saved to: {model_dir}")

try:
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[trial_callback],
        show_progress_bar=True
    )
    
    print("\n[Optuna] Optimization completed successfully!")
    
except KeyboardInterrupt:
    print("\n[Optuna] Optimization interrupted by user")
except Exception as e:
    print(f"\n[Optuna] Optimization failed with error: {e}")

# --- ALWAYS save final study progress regardless of success/failure ---
print("\n[Optuna] Saving final study progress...")
save_study_progress(study)

# --- Evaluate Results ---
print("\n--- Optimization Results ---")
print(f"Number of finished trials: {len(study.trials)}")
print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
print(f"Number of failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

if study.best_trial:
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.6f}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Extract best parameters (similar to original script)
    batchsize_int = study.best_params['batchsize']
    densesize_int = study.best_params['densesize']
    windowsize_int = study.best_params['windowsize']
    filters_int = study.best_params['filters']
    
    print("\n --- Best values: ---")
    print(f"batchsize_int: {batchsize_int}, densesize_int: {densesize_int}, filters_int: {filters_int}, windowsize_int: {windowsize_int}")
    
    # Save only the simple best parameters file (second required file)
    best_params_simple = {
        'timestamp': datetime.datetime.now().isoformat(),
        'best_trial': {
            'number': study.best_trial.number,
            'score': study.best_value,
            'params': {
                'batchsize': batchsize_int,
                'densesize': densesize_int,
                'windowsize': windowsize_int,
                'filters': filters_int
            },
            'marked_as': '🏆 BEST TRIAL 🏆'
        },
        'summary': {
            'n_trials': len(study.trials),
            'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'n_failed': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        }
    }
    
    best_params_file = model_dir / 'best_params.json'
    with open(best_params_file, 'w') as f:
        json.dump(best_params_simple, f, indent=2)
    
    print(f"\n--- Optimization completed ---")
    print(f"All trials log: {model_dir / 'all_trials_log.json'}")
    print(f"Best params: {best_params_file}")

else:
    print("\nNo successful trials found!")

print(f"\n[Optuna] All logs saved to: {model_dir}")
print("[Optuna] Check progress in console output above")

# No temporary files to clean up (using all files directly)

# --- Add entry to master optimization log ---
master_log_file = BASE_DIR / "model_runs" / "OptunaOpt" / "optimization_runs_log.json"
os.makedirs(master_log_file.parent, exist_ok=True)

# Load existing log or create new one
if os.path.exists(master_log_file):
    with open(master_log_file, 'r') as f:
        master_log = json.load(f)
else:
    master_log = {"optimization_runs": []}

# Add this run to the log
if len(study.trials) > 0 and any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
    run_entry = {
        "timestamp": timestamp,
        "directory": model_dir,
        "best_score": study.best_value,
        "best_params": dict(study.best_params),
        "n_trials": len(study.trials),
        "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        "files_used": len(well_ids),
        "optimization_type": "all_files"
    }
else:
    run_entry = {
        "timestamp": timestamp,
        "directory": model_dir,
        "status": "failed - no successful trials",
        "n_trials": len(study.trials),
        "files_used": len(well_ids),
        "optimization_type": "all_files"
    }

master_log["optimization_runs"].append(run_entry)

# Save updated master log
with open(master_log_file, 'w') as f:
    json.dump(master_log, f, indent=2)

print(f"\n📝 Added run to master log: {master_log_file}")
print(f"📁 This run's results: {model_dir}")
print(f"🕐 Total optimization runs logged: {len(master_log['optimization_runs'])}")
