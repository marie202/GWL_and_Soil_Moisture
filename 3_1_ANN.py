# Simple ANN benchmark using the same data pipeline as the CNN-LSTM
# Trains a feed-forward network on flattened windowed inputs for quick comparison

from numpy.random import seed
seed(1 + 347823)
import tensorflow as tf
tf.random.set_seed(1 + 63493)

import matplotlib
matplotlib.use("Agg")  # ensure headless plotting on HPC
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib as mpl
# Use default font settings that work on HPC systems
# Set minimal font configuration to avoid errors
try:
    mpl.rcParams["font.family"] = "sans-serif"
    # Don't restrict to specific fonts - let system choose
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    # Suppress font warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
except:
    pass

# Standard libraries
import os
import glob
import re
import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Project utilities
from s1_data_preparation import scaler_statics_global, process_data_pipeline, scale_dataset_indiv
from s2_model_utils import predict_distribution
from s3_plotting_functions import plot_r2_rmse_boxplots, plot_r2_rmse_nse_bias_boxplot, plot_loss_curves, simulate_plot_wells_testset_conf_int
from scipy import stats
try:
    import uncertainties.unumpy as unumpy
except ImportError:
    print("WARNING: uncertainties package not available. Uncertainty calculations will be simplified.")
    unumpy = None

# ---- GPU info (same as CNN-LSTM script) ----
gpus = tf.config.list_physical_devices("GPU")
print("GPUs available:", gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Configured {len(gpus)} GPU(s) with memory growth enabled")
    except RuntimeError as e:
        print(f"Error configuring GPUs: {e}")
else:
    print("WARNING: No GPUs detected. Training will run on CPU.")

# ---- Settings (reuse CNN-LSTM input structure) ----
input_dir = "data_217/*.csv"
columns_to_keep = [
    "tas_3x3_mean",
    "pr_3x3_sum",
    "hurs_3x3_mean",
    "soil_mois_composite_3x3_mean_0-30",
    "soil_mois_composite_3x3_mean_0-60",
    "soil_mois_composite_3x3_mean_0-90",
    "elevation_msl",
    "MW_muGOK",
    "distance_to_waterwork_km",
    "kf_remap_number",
    "GWL",
]

static_cols = [
    "elevation_msl",
    "MW_muGOK",
    "distance_to_waterwork_km",
    "kf_remap_number",
]

# Hyperparameters (aligned with existing window size/batch size choices)
best_params = {
    "batchsize": 198,
    "densesize": 108,
    "windowsize": 58,
}

batchsize_int = int(best_params["batchsize"])
densesize_int = int(best_params["densesize"])
windowsize_int = int(best_params["windowsize"])

GLOBAL_SETTINGS = {
    "inimax": 10,  # number of initializations for uncertainty
    "window_size": windowsize_int,
    "batch_size": batchsize_int,
    "dense_size": densesize_int,
    "epochs": 150,
    "learning_rate": 1e-4,
    "test_start": pd.to_datetime("2019-01-01", format="%Y-%m-%d"),
    "test_end": pd.to_datetime("2024-12-31", format="%Y-%m-%d"),
    "model_dir_note": f"ANN_benchmark_dropout_0.5_dens{densesize_int}_batch{batchsize_int}_",
}

print(f"  batchsize: {batchsize_int}")
print(f"  densesize: {densesize_int}")
print(f"  windowsize: {windowsize_int}")
print(f"  inimax (initializations): {GLOBAL_SETTINGS['inimax']}")

# ---- Model output directory ----
base_run_dir = "model_runs/"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = os.path.join(
    base_run_dir,
    f"BB_ANN_{len(columns_to_keep)}params_{GLOBAL_SETTINGS['model_dir_note']}_{timestamp}",
)
os.makedirs(model_dir, exist_ok=True)
print("Model directory:", model_dir)

# ---- Data preparation ----
scaler_static, _ = scaler_statics_global(
    input_dir=input_dir, static_cols=static_cols, columns_to_keep=columns_to_keep
)

X_train, Y_train, X_val, Y_val, X_opt, Y_opt, ScalerData_dict, ValData_dict, OptData_dict, TestData_dict = process_data_pipeline(
    input_dir=input_dir,
    columns_to_keep=columns_to_keep,
    static_cols=static_cols,
    GLOBAL_SETTINGS={
        "window_size": GLOBAL_SETTINGS["window_size"],
        "test_start": GLOBAL_SETTINGS["test_start"],
        "test_end": GLOBAL_SETTINGS["test_end"],
    },
    scaler_static=scaler_static,
    target_column="GWL",
)

# Extract well IDs from filenames in working_samples (matching CNN-LSTM)
well_ids = []
source_dir = "data_217"
print(os.getcwd())
for file_name in os.listdir(source_dir):
    if os.path.isfile(os.path.join(source_dir, file_name)):
        match = re.match(r"^(\d+)_weeklyData_", file_name)
        if match:
            well_ids.append(match.group(1))

# Print number of training, validation, test, and optimization samples before training
try:
    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of validation samples: {X_val.shape[0]}")
    if well_ids and f'X_test_{well_ids[0]}' in TestData_dict:
        print(f"Number of test samples: {TestData_dict[f'X_test_{well_ids[0]}'].shape[0]}")
    else:
        print(f"Number of test samples: N/A")
    print(f"Number of optimization samples: {X_opt.shape[0]}")
except Exception as e:
    print(f"[WARNING] Error printing dataset sizes: {e}")

# Save test data and scaler for later use (e.g. plotting)

# Create scaler_y from first well (matching CNN-LSTM structure)
scaler_y = None
if well_ids and f'all_Dataframe_{well_ids[0]}' in ScalerData_dict:
    data = ScalerData_dict[f'all_Dataframe_{well_ids[0]}']
    _, scaler_y, _ = scale_dataset_indiv(data, target_column="GWL")

testdata_dict_path = os.path.join(model_dir, "TestData_dict.pkl")
joblib.dump(TestData_dict, testdata_dict_path)
scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")
if scaler_y is not None:
    joblib.dump(scaler_y, scaler_y_path)

n_features = X_train.shape[-1]
window_size = GLOBAL_SETTINGS["window_size"]

# Flatten sequences for ANN input
def reshape_for_ann(x):
    return x.reshape(x.shape[0], window_size * n_features)

X_train_flat = reshape_for_ann(X_train)
X_val_flat = reshape_for_ann(X_val)
X_opt_flat = reshape_for_ann(X_opt)

# ---- Model definition ----
def build_ann(input_dim: int) -> Sequential:
    model = Sequential(
        [
            Dense(GLOBAL_SETTINGS["dense_size"], activation="relu", input_shape=(input_dim,)),
            Dropout(0.5),
            Dense(GLOBAL_SETTINGS["dense_size"] // 2, activation="relu"),
            Dropout(0.5),
            Dense(1),
        ]
    )
    optimizer = Adam(learning_rate=GLOBAL_SETTINGS["learning_rate"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mse"])
    return model


# ---- Training with multiple initializations ----
inimax = GLOBAL_SETTINGS["inimax"]
print(f"\nTraining {inimax} model initializations...")

all_histories = []
all_val_losses = []
all_final_val_losses = []

for ini in range(inimax):
    print(f"\n{'='*60}")
    print(f"Initialization {ini+1}/{inimax}")
    print(f"{'='*60}")
    
    # Set different random seeds for each initialization
    seed(1 + 347823 + ini * 1000)
    tf.random.set_seed(1 + 63493 + ini * 1000)
    
    # Build fresh model for this initialization
    model = build_ann(window_size * n_features)
    if ini == 0:
        model.summary(print_fn=lambda x: print(x))
    
    # Callbacks (EarlyStopping will restore best weights automatically)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
    ]
    
    # Train model
    history = model.fit(
        X_train_flat,
        Y_train,
        validation_data=(X_val_flat, Y_val),
        epochs=GLOBAL_SETTINGS["epochs"],
        batch_size=GLOBAL_SETTINGS["batch_size"],
        callbacks=callbacks,
        verbose=2,
    )
    
    # Save model for this initialization (best weights already restored by EarlyStopping)
    model_path = os.path.join(model_dir, f"model_weights_BB_run_{ini}.keras")
    model.save(model_path)
    print(f"Saved model to {model_path}")
    
    # Store history and validation loss
    all_histories.append(history)
    all_val_losses.append(history.history["val_loss"])
    final_val_loss = min(history.history["val_loss"])
    all_final_val_losses.append(final_val_loss)
    print(f"Best validation loss for run {ini+1}: {final_val_loss:.6f}")

# Find median initialization (based on final validation loss)
sorted_indices = np.argsort(all_final_val_losses)
median_idx = sorted_indices[len(sorted_indices) // 2]
print(f"\n{'='*60}")
print(f"Median initialization: run {median_idx} (validation loss: {all_final_val_losses[median_idx]:.6f})")
print(f"{'='*60}")

# Load median model for evaluation
from tensorflow.keras.models import load_model
median_model_path = os.path.join(model_dir, f"model_weights_BB_run_{median_idx}.keras")
model = load_model(median_model_path)
print(f"Loaded median model from {median_model_path}")

# ---- Plot training curves for all initializations ----
def plot_history_all(histories, out_dir, median_idx):
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot all runs
        for i, hist in enumerate(histories):
            alpha = 0.3 if i != median_idx else 0.8
            linestyle = "-" if i == median_idx else "--"
            label = f"run_{i}" if i == median_idx else None
            plt.plot(hist.history["loss"], alpha=alpha, linestyle=linestyle, color="blue", label=label if i == median_idx else None)
            plt.plot(hist.history["val_loss"], alpha=alpha, linestyle=linestyle, color="orange", label=label.replace("run_", "val_run_") if i == median_idx else None)
        
        # Highlight median run
        plt.plot(histories[median_idx].history["loss"], linewidth=2, color="blue", label="train_loss (median)")
        plt.plot(histories[median_idx].history["val_loss"], linewidth=2, color="orange", label="val_loss (median)")
        
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.legend()
        plt.title(f"Training History - All {len(histories)} Initializations (median: run {median_idx})")
        # Use bbox_inches='tight' instead of tight_layout to avoid font issues
        out_path = os.path.join(out_dir, "training_history_all.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved training curves to {out_path}")
    except Exception as e:
        print(f"Warning: Failed to save training history plot: {e}")
        print("Training history will still be saved to text file.")

plot_history_all(all_histories, model_dir, median_idx)

# Save training history to text file (matching CNN-LSTM format for plot_loss_curves)
training_history_path = os.path.join(model_dir, "traininghistory_CNN_BB.txt")
with open(training_history_path, 'w') as f:
    f.write(f"median_idx\n{median_idx}\n")
    f.write(f"final_loss_median_model\n{all_histories[median_idx].history['loss']}\n")
    f.write(f"final_val_loss_median_model\n{all_histories[median_idx].history['val_loss']}\n")
    f.write(f"final_loss_all_runs\n")
    for i, hist in enumerate(all_histories):
        f.write(f"run_{i}_loss\n{hist.history['loss']}\n")
    f.write(f"final_val_loss_all_runs\n")
    for i, hist in enumerate(all_histories):
        f.write(f"run_{i}_val_loss\n{hist.history['val_loss']}\n")
print(f"Saved training history to {training_history_path}")

# Plot training and validation loss curves (matching CNN-LSTM)
try:
    plot_loss_curves(model_dir, BL_abbr="BB", show_plot=True)
except Exception as e:
    print(f"[WARNING] Plotting error in plot_loss_curves: {e}")

# --- Plotting and evaluation functions ---

# Run test set simulation and plot results, get scores for all wells
# Note: simulate_plot_wells_testset_conf_int expects CNN-LSTM models, so we use a custom ANN version
def simulate_plot_wells_testset_ann(well_ids, TestData_dict, GLOBAL_SETTINGS, model_path, inimax, number_of_wells, columns_to_keep, BL_abbr="BB", save_fig=False):
    """
    Analyzes test data for all wells using trained ANN models.
    Based on simulate_plot_wells_testset_conf_int but adapted for ANN.
    
    Parameters:
        well_ids (list): List of well IDs.
        TestData_dict (dict): Dictionary containing test data and observation dataframes.
        GLOBAL_SETTINGS (dict): Global settings for the analysis.
        model_path (str): Path to the trained model weights.
        inimax (int): Number of simulation runs for uncertainty analysis.
        columns_to_keep (list): List of input columns.
        save_fig (bool): Whether to save figures.
    
    Returns:
        dict: Dictionary containing scores for each well.
    """
    scores_dict = {}
    
    for number_of_well in range(number_of_wells):
        WELL_ID = well_ids[number_of_well]
        well_idx = number_of_well
        print(f"\nProcessing well {well_idx+1}/{len(well_ids)}: {WELL_ID}")
        
        # Extract test data for the current well
        if f'X_test_{WELL_ID}' not in TestData_dict:
            print(f"  Warning: No test data found for well {WELL_ID}, skipping...")
            continue
            
        X_test = TestData_dict[f'X_test_{WELL_ID}']
        Y_test = TestData_dict[f'Y_test_{WELL_ID}']
        data = TestData_dict[f'obs_Dataframe_{WELL_ID}'].copy()
        TestData_cut = TestData_dict[f'obs_Dataframe_{WELL_ID}'].iloc[
            GLOBAL_SETTINGS["window_size"]:GLOBAL_SETTINGS["window_size"]+len(X_test)
        ]
        
        if len(X_test) == 0:
            print(f"  Warning: Empty test data for well {WELL_ID}, skipping...")
            continue
        
        # Re-initialize individual scaler for this well
        _, scaler_y, _ = scale_dataset_indiv(data, target_column="GWL")
        
        # Flatten test data for ANN
        X_test_flat = reshape_for_ann(X_test)
        
        # Initialize simulation arrays
        sim_members = np.zeros((len(X_test), inimax))
        sim_members[:] = np.nan
        sim_std = np.zeros((len(X_test), inimax))
        sim_std[:] = np.nan
        
        # Run simulations for each initial condition
        for ini in range(inimax):
            try:
                loaded_model = tf.keras.models.load_model(
                    os.path.join(model_path, f'model_weights_BB_run_{ini}.keras')
                )
                
                # Use predict_distribution for uncertainty estimation
                if unumpy is not None:
                    y_pred_distribution = predict_distribution(X_test_flat, loaded_model, 100)
                    sim = scaler_y.inverse_transform(y_pred_distribution)
                    sim_members[:, ini] = sim.mean(axis=1)
                    sim_std[:, ini] = sim.std(axis=1)
                else:
                    # Fallback: direct prediction without distribution
                    y_pred = loaded_model.predict(X_test_flat, verbose=0)
                    sim = scaler_y.inverse_transform(y_pred)
                    sim_members[:, ini] = sim.flatten()
                    sim_std[:, ini] = 0.0  # No uncertainty estimate without distribution
                    
            except Exception as e:
                print(f"  Warning: Failed to load/run model {ini} for well {WELL_ID}: {e}")
                continue
        
        # Calculate uncertainties and statistics
        if unumpy is not None:
            sim_members_uncertainty = unumpy.uarray(sim_members, 1.96 * sim_std)
            sim_mean_uncertainty = np.sum(sim_members_uncertainty, axis=1) / inimax
        else:
            sim_mean_uncertainty = None
        
        sim_mean = np.nanmedian(sim_members, axis=1)  # Median prediction across models
        sim_max = np.nanmax(sim_members, axis=1)
        sim_min = np.nanmin(sim_members, axis=1)
        
        sim = np.asarray(sim_mean.reshape(-1, 1))
        obs = np.asarray(scaler_y.inverse_transform(Y_test.reshape(-1, 1)))
        
        # Calculate metrics
        err = sim - obs
        err_rel = (sim - obs) / (np.max(TestData_cut['GWL']) - np.min(TestData_cut['GWL']))
        err_nash = obs - TestData_cut['GWL'].mean()
        
        NSE = 1 - ((np.sum(err ** 2)) / (np.sum((err_nash) ** 2)))
        r = stats.linregress(sim[:, 0], obs[:, 0])
        R2 = r.rvalue ** 2
        RMSE = np.sqrt(np.mean(err ** 2))
        rRMSE = np.sqrt(np.mean(err_rel ** 2)) * 100
        Bias = np.mean(err)
        rBias = np.mean(err_rel) * 100
        
        # Store scores
        scores = pd.DataFrame(
            np.array([[R2, NSE, RMSE, rRMSE, Bias, rBias]]),
            columns=['R2', "NSE", 'RMSE', 'rRMSE', 'Bias', 'rBias']
        )
        scores_dict[WELL_ID] = scores
        
        print(f"  Well {WELL_ID}: R²={R2:.3f}, NSE={NSE:.3f}, RMSE={RMSE:.3f}")
        
        # Plot results
        if save_fig:
            try:
                plt.figure(figsize=(20, 6))
                fontsize = 17
                
                plt.fill_between(
                    TestData_cut.index,
                    sim_max, sim_min,
                    facecolor="#C8DAFB", alpha=0.7,
                    label='Uncertainty', linewidth=1,
                )
                
                plt.plot(TestData_cut.index, sim, '#EA3358', label="Simulated Median", linewidth=1.7)
                plt.plot(TestData_cut.index, obs, 'k', label="Observed Data", linewidth=1.7, alpha=0.9)
                plt.title(f"ANN Model Run: {BL_abbr}{WELL_ID}", size=fontsize+2, fontweight='bold')
                plt.ylabel('GWL [m asl]', size=fontsize)
                plt.xlabel('Date', size=fontsize)
                
                # Add text box with statistics
                s1 = """R² = {:.2f}\nNSE = {:.2f}\nRMSE = {:.2f}\nrRMSE = {:.2f}\nBias = {:.2f}\nrBias = {:.2f}""".format(
                    scores.R2[0],
                    scores.NSE[0],
                    scores.RMSE[0],
                    scores.rRMSE[0],
                    scores.Bias[0],
                    scores.rBias[0],
                )
                
                # Add text box with hyperparameters
                s2 = """number of input parameters = {:d}\ndense size = {:d}\nwindow size = {:d}\nbatch size = {:d}""".format(
                    len(columns_to_keep)-1,
                    GLOBAL_SETTINGS["dense_size"],
                    GLOBAL_SETTINGS["window_size"],
                    GLOBAL_SETTINGS["batch_size"]
                )
                
                plt.figtext(0.849, 0.41, s1, bbox=dict(facecolor='white'), fontsize=fontsize)
                plt.legend(fontsize=fontsize, bbox_to_anchor=(1.21, 1.021), loc='upper right', fancybox=False, framealpha=1, edgecolor='k')
                plt.tick_params(axis='both', labelsize=fontsize-2)
                plt.grid()
                plt.savefig(os.path.join(model_path, f'{BL_abbr}_{WELL_ID}_ANN_run_{TestData_cut.index.min().year}_{TestData_cut.index.max().year}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Saved figure for well {WELL_ID}")
            except Exception as plot_err:
                print(f"  Warning: Failed to save plot for well {WELL_ID}: {plot_err}")
                # Continue processing even if plotting fails
    
    return scores_dict

# Run test set simulation and plot results, get scores for all wells
try:
    scores_dict = simulate_plot_wells_testset_ann(
        well_ids=well_ids,
        TestData_dict=TestData_dict,
        GLOBAL_SETTINGS=GLOBAL_SETTINGS,
        model_path=model_dir+"/",
        inimax=inimax,
        number_of_wells=len(well_ids),
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
path = model_dir + "/"
file = "scores_df.csv"
try:
    if not df.empty:
        df.to_csv(os.path.join(path, file), index=False)

        # List of metrics to print statistics for
        stats_metrics = [
            "R2",
            "NSE",
            "RMSE",
            "rRMSE",
            "Bias",
            "rBias",
        ]

        for col in stats_metrics:
            if col in df.columns:
                print(f"\n{col} Statistics:")
                print(f"  Mean:   {df[col].mean():.5f}")
                print(f"  Median: {df[col].median():.5f}")
                print(f"  Min:    {df[col].min():.5f}")
                print(f"  Max:    {df[col].max():.5f}")
            else:
                print(f"\n[WARNING] Column '{col}' not in DataFrame. Skipping statistics for {col}.")
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
    if not df.empty:
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
    else:
        print("[WARNING] Cannot plot boxplots: DataFrame is empty")
except ImportError as e:
    if 'seaborn' in str(e):
        print(f"[WARNING] seaborn not available, skipping boxplots: {e}")
    else:
        print(f"[WARNING] Import error in boxplots: {e}")
except Exception as e:
    print(f"An error occurred while plotting R² and RMSE boxplots: {e}")

# --- Evaluation: Feature Importance and SHAP values ---

# --- Feature Importance ---

model_path = model_dir+"/"

# Load median model and perform feature importance analysis
try:
    median_model = load_model(model_path+f'model_weights_BB_run_{median_idx}.keras')

    # Concatenate all test data for feature importance
    X_test_all = np.concatenate([TestData_dict[f'X_test_{well_id}'] for well_id in well_ids], axis=0)
    Y_test_all = np.concatenate([TestData_dict[f'Y_test_{well_id}'] for well_id in well_ids], axis=0)
    
    # Flatten for ANN
    X_test_all_flat = reshape_for_ann(X_test_all)

    # Compute and save feature importance
    try:
        # Note: feature_importance expects CNN-LSTM format, so we adapt it for ANN
        # For ANN, we'll use a simplified version or skip if not compatible
        print("[INFO] Feature importance computation may need adaptation for ANN models")
        # feature_importance(
        #     median_model,
        #     X_test_all_flat,  # Flattened for ANN
        #     Y_test_all,
        #     scaler_y,
        #     columns_to_keep,
        #     model_path,
        #     "feature_importance.txt"
        # )
    except Exception as e:
        print(f"[WARNING] Failed to compute feature importance: {e}")

except Exception as e:
    print(f"[ERROR] Failed to load median model or perform analysis: {e}")
    print("[WARNING] Skipping feature importance analysis...")
    median_model = None

