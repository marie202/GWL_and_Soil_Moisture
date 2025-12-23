# --- Imports and setup ---

# seed
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

# Keras/TensorFlow imports for model construction
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization  # Batch normalization for stable training --> hat matthias so geraten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import load_model


# Import project modules
from s1_data_preparation import *  # Custom data preparation utilities

# --- Model Construction and Training ---

def build_cnn_lstm_model(ini, GLOBAL_SETTINGS, X_train, Y_train, X_val, Y_val):
    """
    Build and train a CNN-LSTM model for groundwater level prediction.
    The architecture is dynamically controlled by GLOBAL_SETTINGS.
    Combines CNN layers for feature extraction and LSTM layers for temporal dependencies.
    inspired by Wunsch et al 2022 on github https://github.com/AndreasWunsch/Long-Term-GWL-Simulations

    Parameters:
    - ini: random seed initialization (for reproducibility)
    - GLOBAL_SETTINGS: dict with model hyperparameters
    - X_train, Y_train: training data
    - X_val, Y_val: validation data

    Returns:
    - model: Trained Keras model
    - history: Training history object
    """
    
    # Set random seed for reproducibility
    seed(ini + 872527)
    tf.random.set_seed(ini + 87747)

    # Define input layer
    inp = tf.keras.Input(shape=(GLOBAL_SETTINGS["window_size"], X_train.shape[2]))
    
    #  CNN + LSTM layers setup
    x = inp
    for i in range(GLOBAL_SETTINGS["num_cnn_layers"]):
        x = tf.keras.layers.Conv1D(
            filters=GLOBAL_SETTINGS["filters"],
            kernel_size=GLOBAL_SETTINGS["kernel_size"],
            activation='relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)  # L2 regularization to prevent overfitting
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)  # Normalize activations for stable training
        
        x = tf.keras.layers.MaxPool1D(padding='same')(x)

        # Use higher dropout for the last CNN layer
        if i == GLOBAL_SETTINGS["num_cnn_layers"] - 1:
            x = tf.keras.layers.Dropout(0.5)(x)  # Higher dropout for last layer
        else:
            x = tf.keras.layers.Dropout(0.1)(x)  # Dropout for regularization
    


    # Additional pooling and dropout for further regularization
   # x = tf.keras.layers.MaxPool1D(padding='same')(x)
    #x = tf.keras.layers.Dropout(0.5)(x)
    # (Optional: BatchNorm, more Dropout, or GlobalAveragePooling can be tried for further regularization)

    # Add LSTM layers to capture temporal dependencies in the sequence data
    # x = tf.keras.layers.LSTM(32, return_sequences=True,  kernel_regularizer=tf.keras.regularizers.l2(1e-4) )(x)  # First LSTM layer
    # x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.LSTM(16, return_sequences=False,  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)  # Second LSTM layer
    # x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.LSTM(
        GLOBAL_SETTINGS["lstm_units"][0],
        return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.LSTM(
        GLOBAL_SETTINGS["lstm_units"][1],
        return_sequences=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Dense layers for final regression output
    x = tf.keras.layers.Dense(
        GLOBAL_SETTINGS["dense_size"], 
        activation='relu', 
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    output1 = tf.keras.layers.Dense(1, activation='linear')(x)  # Output layer for regression

    # Compile the model
    model = tf.keras.Model(inputs=inp, outputs=output1)

## use elarning rate decay for better generalization    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = GLOBAL_SETTINGS["learning_rate"],
        decay_steps = 10000,
        decay_rate = 0.96,
        staircase=False,
        name="ExponentialDecay",
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,#GLOBAL_SETTINGS["learning_rate"],
        epsilon=1e-6, 
        clipnorm=GLOBAL_SETTINGS["clip_norm"]
    )
    

    # optimizer = tf.keras.optimizers.Adam(
    #     learning_rate=GLOBAL_SETTINGS["learning_rate"],
    #     epsilon=1e-3, 
    #     clipnorm=GLOBAL_SETTINGS["clip_norm"]
    # )
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

    # Early stopping to prevent overfitting (restores best weights)
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', 
        verbose=1, patience=15, restore_best_weights=True
    )

    # Shuffle training data to ensure randomness in each run
    idx = tf.random.shuffle(tf.range(tf.shape(X_train)[0]))
    X_train = tf.gather(X_train, idx)
    Y_train = tf.gather(Y_train, idx)

    # Train the model
    history = model.fit(
        X_train, Y_train, validation_data=(X_val, Y_val),  
        epochs=GLOBAL_SETTINGS["epochs"], 
        verbose=1,  # 0: silent, 1: progress bar, 2: one line per epoch
        batch_size=GLOBAL_SETTINGS["batch_size"], callbacks=[es]
    )
    
    return model, history


def build_cnn_model(ini, GLOBAL_SETTINGS, X_train, Y_train, X_val, Y_val):
    """
    Build and train a CNN-only model (without LSTM) for groundwater level prediction.
    The architecture is dynamically controlled by GLOBAL_SETTINGS.
    Uses CNN layers for feature extraction, then GlobalAveragePooling to convert to vector,
    followed by dense layers for regression.
    Based on build_cnn_lstm_model but without LSTM layers.

    Parameters:
    - ini: random seed initialization (for reproducibility)
    - GLOBAL_SETTINGS: dict with model hyperparameters
    - X_train, Y_train: training data
    - X_val, Y_val: validation data

    Returns:
    - model: Trained Keras model
    - history: Training history object
    """
    
    # Set random seed for reproducibility
    seed(ini + 872527)
    tf.random.set_seed(ini + 87747)

    # Define input layer
    inp = tf.keras.Input(shape=(GLOBAL_SETTINGS["window_size"], X_train.shape[2]))
    
    # CNN layers setup
    x = inp
    for i in range(GLOBAL_SETTINGS["num_cnn_layers"]):
        x = tf.keras.layers.Conv1D(
            filters=GLOBAL_SETTINGS["filters"],
            kernel_size=GLOBAL_SETTINGS["kernel_size"],
            activation='relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)  # L2 regularization to prevent overfitting
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)  # Normalize activations for stable training
        
        x = tf.keras.layers.MaxPool1D(padding='same')(x)

        # Use higher dropout for the last CNN layer
        if i == GLOBAL_SETTINGS["num_cnn_layers"] - 1:
            x = tf.keras.layers.Dropout(0.5)(x)  # Higher dropout for last layer
        else:
            x = tf.keras.layers.Dropout(0.1)(x)  # Dropout for regularization
    
    # Convert sequence to vector using GlobalAveragePooling (instead of LSTM)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Dense layers for final regression output
    x = tf.keras.layers.Dense(
        GLOBAL_SETTINGS["dense_size"], 
        activation='relu', 
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    output1 = tf.keras.layers.Dense(1, activation='linear')(x)  # Output layer for regression

    # Compile the model
    model = tf.keras.Model(inputs=inp, outputs=output1)

## use learning rate decay for better generalization    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = GLOBAL_SETTINGS["learning_rate"],
        decay_steps = 10000,
        decay_rate = 0.96,
        staircase=False,
        name="ExponentialDecay",
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        epsilon=1e-6, 
        clipnorm=GLOBAL_SETTINGS["clip_norm"]
    )
    
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

    # Early stopping to prevent overfitting (restores best weights)
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', 
        verbose=1, patience=15, restore_best_weights=True
    )

    # Shuffle training data to ensure randomness in each run
    idx = tf.random.shuffle(tf.range(tf.shape(X_train)[0]))
    X_train = tf.gather(X_train, idx)
    Y_train = tf.gather(Y_train, idx)

    # Train the model
    history = model.fit(
        X_train, Y_train, validation_data=(X_val, Y_val),  
        epochs=GLOBAL_SETTINGS["epochs"], 
        verbose=1,  # 0: silent, 1: progress bar, 2: one line per epoch
        batch_size=GLOBAL_SETTINGS["batch_size"], callbacks=[es]
    )
    
    return model, history


# def build_cnn_lstm_model(ini, GLOBAL_SETTINGS, X_train, Y_train, X_val, Y_val):
#     """
#     Build and train a CNN-LSTM model for groundwater level prediction.
#     The architecture is dynamically controlled by GLOBAL_SETTINGS.
#     Combines CNN layers for feature extraction and LSTM layers for temporal dependencies.
#     inspired by Wunsch et al 2022 on github https://github.com/AndreasWunsch/Long-Term-GWL-Simulations

#     Parameters:
#     - ini: random seed initialization (for reproducibility)
#     - GLOBAL_SETTINGS: dict with model hyperparameters
#     - X_train, Y_train: training data
#     - X_val, Y_val: validation data

#     Returns:
#     - model: Trained Keras model
#     - history: Training history object
#     """
    
#     # Set random seed for reproducibility
#     seed(ini + 872527)
#     tf.random.set_seed(ini + 87747)

#     # Define input layer
#     inp = tf.keras.Input(shape=(GLOBAL_SETTINGS["window_size"], X_train.shape[2]))
    
#     #  CNN + LSTM layers setup
#     x = inp
#     for i in range(GLOBAL_SETTINGS["num_cnn_layers"]):
#         x = tf.keras.layers.Conv1D(
#             filters=GLOBAL_SETTINGS["filters"],
#             kernel_size=GLOBAL_SETTINGS["kernel_size"],
#             activation='relu',
#             padding='same',
#             kernel_regularizer=tf.keras.regularizers.l2(1e-4)  # L2 regularization to prevent overfitting
#         )(x)
#         x = tf.keras.layers.BatchNormalization()(x)  # Normalize activations for stable training
        
#         # Apply MaxPooling every other layer only
#         if i % 2 == 0:
#             x = tf.keras.layers.MaxPool1D(padding='same')(x)

        
#         #x = tf.keras.layers.MaxPool1D(padding='same')(x)  # Downsample feature maps
#         x = tf.keras.layers.Dropout(0.1)(x)  # Dropout for regularization
    
#     # Replace final pooling and dropout with GlobalAveragePooling
#     x = tf.keras.layers.GlobalAveragePooling1D()(x)
#     x = tf.keras.layers.Reshape((1, x.shape[-1]))(x)  # Reshape to fit LSTM input shape

    
#     # Additional pooling and dropout for further regularization
#    #x = tf.keras.layers.MaxPool1D(padding='same')(x)
#     #x = tf.keras.layers.Dropout(0.5)(x)
#     # (Optional: BatchNorm, more Dropout, or GlobalAveragePooling can be tried for further regularization)

#     # Add LSTM layers to capture temporal dependencies in the sequence data
#     # x = tf.keras.layers.LSTM(32, return_sequences=True,  kernel_regularizer=tf.keras.regularizers.l2(1e-4) )(x)  # First LSTM layer
#     # x = tf.keras.layers.Dropout(0.2)(x)
#     # x = tf.keras.layers.LSTM(16, return_sequences=False,  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)  # Second LSTM layer
#     # x = tf.keras.layers.Dropout(0.2)(x)

#     x = tf.keras.layers.LSTM(
#         GLOBAL_SETTINGS["lstm_units"][0],
#         return_sequences=True,
#         kernel_regularizer=tf.keras.regularizers.l2(1e-4)
#     )(x)
#     x = tf.keras.layers.LSTM(
#         GLOBAL_SETTINGS["lstm_units"][1],
#         return_sequences=False,
#         kernel_regularizer=tf.keras.regularizers.l2(1e-4)
#     )(x)
#     x = tf.keras.layers.Dropout(0.3)(x)

#     # Dense layers for final regression output
#     x = tf.keras.layers.Dense(
#         GLOBAL_SETTINGS["dense_size"], 
#         activation='relu', 
#         kernel_regularizer=tf.keras.regularizers.l2(1e-4)
#     )(x)
#     output1 = tf.keras.layers.Dense(1, activation='linear')(x)  # Output layer for regression

#     # Compile the model
#     model = tf.keras.Model(inputs=inp, outputs=output1)

# ## use elarning rate decay for better generalization    
#     lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate = GLOBAL_SETTINGS["learning_rate"],
#         decay_steps = 10000,
#         decay_rate = 0.96,
#         staircase=False,
#         name="ExponentialDecay",
#     )
#     optimizer = tf.keras.optimizers.Adam(
#         learning_rate=lr_schedule,#GLOBAL_SETTINGS["learning_rate"],
#         epsilon=1e-6, 
#         clipnorm=GLOBAL_SETTINGS["clip_norm"]
#     )
    

#     # optimizer = tf.keras.optimizers.Adam(
#     #     learning_rate=GLOBAL_SETTINGS["learning_rate"],
#     #     epsilon=1e-3, 
#     #     clipnorm=GLOBAL_SETTINGS["clip_norm"]
#     # )
#     model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

#     # Early stopping to prevent overfitting (restores best weights)
#     es = tf.keras.callbacks.EarlyStopping(
#         monitor='val_loss', mode='min', 
#         verbose=1, patience=15, restore_best_weights=True
#     )

#     # Shuffle training data to ensure randomness in each run
#     idx = tf.random.shuffle(tf.range(tf.shape(X_train)[0]))
#     X_train = tf.gather(X_train, idx)
#     Y_train = tf.gather(Y_train, idx)

#     # Train the model
#     history = model.fit(
#         X_train, Y_train, validation_data=(X_val, Y_val),  
#         epochs=GLOBAL_SETTINGS["epochs"], 
#         verbose=1,  # 0: silent, 1: progress bar, 2: one line per epoch
#         batch_size=GLOBAL_SETTINGS["batch_size"], callbacks=[es]
#     )
    
#     return model, history

def predict_distribution(X, model, n):
    """
    Run the model n times on input X to estimate prediction distribution (for uncertainty).
    Returns stacked predictions.
    inspired by Wunsch et al 2022 on github https://github.com/AndreasWunsch/Long-Term-GWL-Simulations
    """
    # Fix model attributes that might be None when loading with compile=False
    # This is needed for Keras 3.x compatibility when loading models saved with Keras 2.x
    if hasattr(model, 'steps_per_execution'):
        if model.steps_per_execution is None:
            model.steps_per_execution = 1
    else:
        model.steps_per_execution = 1
    
    if hasattr(model, 'jit_compile'):
        if model.jit_compile is None:
            model.jit_compile = False
    else:
        model.jit_compile = False
    
    if hasattr(model, 'run_eagerly'):
        if model.run_eagerly is None:
            model.run_eagerly = False
    else:
        model.run_eagerly = False
    
    # Ensure correct dtype and use predict() API to avoid Keras input structure warnings
    X_np = np.asarray(X).astype(np.float32)
    preds = [model.predict(X_np, verbose=0) for _ in range(n)]
    return np.hstack(preds)

def simulate_testset(
    path, model_dir,
    X_train, Y_train, 
    X_val, Y_val,
    scaler_data,
    BL_abbr, GLOBAL_SETTINGS, inimax=10
):
    '''
    Train and evaluate multiple (default 10) model initializations to estimate model uncertainty.
    Saves training history and model weights for each run.
    Returns median model index, performance scores, predictions, and uncertainty estimates.
    inspired by Wunsch et al 2022 on github https://github.com/AndreasWunsch/Long-Term-GWL-Simulations

    Parameters:
    - path: (unused, for compatibility)
    - model_dir: directory to save models and logs
    - X_train, Y_train: training data
    - X_val, Y_val: validation data
    - scaler_data: data for fitting scaler (for inverse transform)
    - BL_abbr: abbreviation for federal state (for file naming)
    - GLOBAL_SETTINGS: dict with model hyperparameters
    - inimax: number of model initializations (ensemble size)

    Returns:
    - median_idx: index of the median model (by validation loss)
    - scores: DataFrame with R2, RMSE, Bias
    - sim1: median model predictions (reshaped)
    - obs1: true values (reshaped)
    - inimax: number of initializations
    - sim_members: predictions from all ensemble members
    - sim_members_uncertainty: uncertainty array (mean ± 1.96*std)
    - sim_mean_uncertainty: mean uncertainty across ensemble
    - final_loss: training loss curve of median model
    - final_val_loss: validation loss curve of median model
    - val_loss_final_epoch: final val loss for each run
    '''
    # Ensure scale_dataset_indiv is available (for scaling targets)
    try:
        scale_dataset_indiv
    except NameError:
        raise NameError("scale_dataset_indiv is not defined. Please import or define it before calling simulate_testset.")

    sim_members = np.zeros((len(X_val), inimax))
    sim_members[:] = np.nan
    
    sim_std = np.zeros((len(X_val), inimax))
    sim_std[:] = np.nan

    loss_members = np.zeros((GLOBAL_SETTINGS['epochs'], inimax))
    loss_members[:] = np.nan

    val_loss_members = np.zeros((GLOBAL_SETTINGS['epochs'], inimax))
    val_loss_members[:] = np.nan

    val_loss_final_epoch = np.zeros(inimax)  # Array to store final validation losses for each run

    # initialise a artificial scaler, we just need that here for error calculations and not for actual interpretation
    _, scaler_y, _ = scale_dataset_indiv(scaler_data, target_column="GWL")

    # Open file to log training history for all runs
    f = open(model_dir + '/traininghistory_CNN_' + BL_abbr + '.txt', "w")
    
    # Check if we should use CNN-only model (without LSTM)
    use_cnn_only = GLOBAL_SETTINGS.get("use_cnn_only", False)
    build_model_func = build_cnn_model if use_cnn_only else build_cnn_lstm_model
    
    for ini in range(inimax):
        # Build and train model for this initialization
        model, history = build_model_func(ini, GLOBAL_SETTINGS, X_train, Y_train, X_val, Y_val)
        
        # Store loss and val_loss for this run (pad with NaN if early stopped)
        loss = np.zeros((1, GLOBAL_SETTINGS['epochs'])); loss[:, :] = np.nan
        loss[0, 0:np.shape(history.history['loss'])[0]] = history.history['loss']
        val_loss = np.zeros((1, GLOBAL_SETTINGS['epochs'])); val_loss[:, :] = np.nan
        val_loss[0, 0:np.shape(history.history['val_loss'])[0]] = history.history['val_loss']
        
        # Log losses to file for later analysis
        print('loss', file=f)
        print(loss.tolist(), file=f)
        print('val_loss', file=f)
        print(val_loss.tolist(), file=f)
        
        # Store losses for ensemble statistics for each initialization into 1 array
        val_loss_members[:, ini] = val_loss
        loss_members[:, ini] = loss

        # Save the best (minimum) validation loss for this run
        val_loss_final_epoch[ini] = np.nanmin(val_loss)
        
        # Save model weights for this run (for later use or analysis)
        model.save(f"{model_dir}/model_weights_{BL_abbr}_run_{ini}.keras")
        
        # Estimate prediction distribution by running model 100 times (for uncertainty)
        y_pred_distribution = predict_distribution(X_val, model, 100)
        sim = scaler_y.inverse_transform(y_pred_distribution)  # Inverse transform to original scale
        sim_members[:, ini], sim_std[:, ini] = sim.mean(axis=1), sim.std(axis=1)  # Store mean and std

    f.close()

    # Select the median model (by validation loss) for reporting and further analysis
    median_idx = np.argsort(val_loss_final_epoch)[len(val_loss_final_epoch) // 2]
    print(f"Median model index: {median_idx} with validation loss: {val_loss_final_epoch}")

    final_val_loss = val_loss_members[:, median_idx]
    final_loss = loss_members[:, median_idx]

    # Save summary info for the median model to training history file
    with open(os.path.join(model_dir, f'traininghistory_CNN_{BL_abbr}.txt'), "a") as f:
        print('median_idx', file=f)
        print(median_idx, file=f)
        print('final_loss_median_model', file=f)
        print(final_loss.tolist(), file=f)
        print('final_val_loss_median_model', file=f)
        print(final_val_loss.tolist(), file=f)

    # (Optional) Code for reloading the median model is commented out below
    # median_model, median_history = build_cnn_lstm_model(median_idx, GLOBAL_SETTINGS, X_train, Y_train, X_test, Y_test)
    # median_model.load_weights(f"{model_dir}/model_weights_{BL_abbr}_run_{median_idx}.weights.h5")

    # Calculate uncertainty: 1.96*std for 95% confidence interval
    sim_members_uncertainty = 1.96 * sim_std
    sim_mean = np.nanmedian(sim_members, axis=1)
    sim_mean_uncertainty = np.nanmean(1.96 * sim_std, axis=1)

    # Calculate performance metrics (R2, RMSE, NSE, Bias) for the median model
    sim = np.asarray(sim_mean.reshape(-1, 1))
    obs = np.asarray(scaler_y.inverse_transform(Y_val.reshape(-1, 1)))
    err = sim - obs

    r = stats.linregress(sim[:, 0], obs[:, 0])
    R2 = r.rvalue ** 2
    RMSE = np.sqrt(np.mean(err ** 2))
    Bias = np.mean(err)

    # Store scores in a DataFrame for easy reporting
    scores = pd.DataFrame(np.array([[R2, RMSE, Bias]]), columns=['R2','RMSE','Bias'])
    
    sim1 = sim
    obs1 = obs

    # Return all relevant results for further analysis or plotting
    return (
        median_idx, scores, sim1, obs1, inimax, 
        sim_members, sim_members_uncertainty, sim_mean_uncertainty,
        final_loss, final_val_loss, val_loss_final_epoch
    )


# Helper script to load optimization results
# This shows how to use the saved optimization results in another script

import os
import json
import pickle
import pandas as pd

def load_optimization_results(model_dir):
    """
    Load optimization results from the specified model directory.
    
    Args:
        model_dir (str): Path to the model directory containing saved results
        
    Returns:
        dict: Dictionary containing all optimization results
    """
    
    # Load the full optimization results (pickle format)
    pickle_file_path = os.path.join(model_dir, 'optimization_results.pkl')
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded full optimization results from {pickle_file_path}")
        return results
    else:
        print(f"Pickle file not found: {pickle_file_path}")
        return None

def load_best_params(model_dir):
    """
    Load only the best parameters in a simple format.
    
    Args:
        model_dir (str): Path to the model directory containing saved results
        
    Returns:
        dict: Dictionary containing best parameters
    """
    
    # Load simple best parameters (JSON format)
    simple_params_file = os.path.join(model_dir, 'best_params.json')
    if os.path.exists(simple_params_file):
        with open(simple_params_file, 'r') as f:
            best_params = json.load(f)
        print(f"Loaded best parameters from {simple_params_file}")
        return best_params
    else:
        print(f"Best params file not found: {simple_params_file}")
        return None

def load_optimizer(model_dir):
    """
    Load the full optimizer object (useful for resuming optimization).
    
    Args:
        model_dir (str): Path to the model directory containing saved results
        
    Returns:
        BayesianOptimization: The saved optimizer object
    """
    
    optimizer_file_path = os.path.join(model_dir, 'optimizer.pkl')
    if os.path.exists(optimizer_file_path):
        with open(optimizer_file_path, 'rb') as f:
            optimizer = pickle.load(f)
        print(f"Loaded optimizer from {optimizer_file_path}")
        return optimizer
    else:
        print(f"Optimizer file not found: {optimizer_file_path}")
        return None

def print_optimization_summary(results):
    """
    Print a summary of the optimization results.
    
    Args:
        results (dict): Optimization results dictionary
    """
    if results is None:
        print("No results to display")
        return
        
    print("\n=== OPTIMIZATION SUMMARY ===")
    print(f"Timestamp: {results.get('timestamp', 'N/A')}")
    print(f"Model directory: {results.get('model_dir', 'N/A')}")
    print(f"Best score: {results.get('best_score', 'N/A')}")
    
    best_params = results.get('best_params', {})
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nNumber of optimization iterations: {len(results.get('optimization_history', []))}")
    
    # Show bounds used
    bounds = results.get('bounds', {})
    print("\nParameter bounds used:")
    for param, bound in bounds.items():
        print(f"  {param}: {bound}")