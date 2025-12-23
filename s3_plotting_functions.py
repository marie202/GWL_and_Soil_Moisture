# --- Imports and setup ---
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set robust font fallbacks for HPC environments where DejaVu may be missing
mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.sans-serif'] = [
    'Liberation Sans', 'Arial', 'Helvetica', 'Nimbus Sans', 'FreeSans', 'DejaVu Sans'
]
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# import seaborn as sns
import pandas as pd
import shap  # For SHAP values

# Workaround for Python 3.12+ zipfile cp437 encoding issue
import zipfile
try:
    import codecs
    codecs.lookup('cp437')
except LookupError:
    # If cp437 not available, use utf-8
    zipfile.ZipFile._metadata_encoding = lambda self: 'utf-8'

# Import project modules
from s1_data_preparation import *  # Custom data preparation utilities
from s2_model_utils import *  # Custom model utilities

# ============================================================================
# Keras 2.x to 3.x Compatibility Fixes
# ============================================================================
# These patches handle compatibility issues when loading models saved with
# Keras 2.x (tf_keras) in an environment with Keras 3.x (keras)
# ============================================================================

import sys
import importlib
import tensorflow as tf

# Import keras first to ensure it's available
import keras

# Fix for tf_keras module mapping issue
# Models saved with tf_keras need to be mapped to keras for loading
if 'tf_keras' not in sys.modules:
    # Store original import_module
    _original_import_module = importlib.import_module
    
    def _patched_import_module(name, package=None):
        """Patched import_module that redirects tf_keras imports to keras and handles functional module"""
        # Handle tf_keras imports
        if name and name.startswith('tf_keras'):
            # Map tf_keras to keras
            keras_name = name.replace('tf_keras', 'keras', 1)
            try:
                # Try to import the keras module
                module = _original_import_module(keras_name, package)
                # Register it under the tf_keras name
                sys.modules[name] = module
                return module
            except (ImportError, ModuleNotFoundError):
                # If it's a functional module, handle it specially below
                if name == 'tf_keras.src.engine.functional':
                    # Will be handled by the functional module handler below
                    # Map to keras.src.engine.functional for unified handling
                    name = 'keras.src.engine.functional'
                else:
                    # For other tf_keras modules, re-raise the error
                    raise
        
        # Handle keras.src.engine.functional imports that might fail
        # (this happens after tf_keras is mapped to keras by the serialization patch)
        # Note: Keras 3.x uses keras.src.models.functional, while tf_keras/Keras 2.x uses keras.src.engine.functional
        # We need to map keras.src.engine.functional -> keras.src.models.functional for Keras 3.x compatibility
        if name == 'keras.src.engine.functional':
            try:
                import keras
                # In Keras 3.x, functional is in models, not engine
                # Try keras.src.models.functional first (Keras 3.x)
                try:
                    from keras.src.models import functional as func_module
                    # Register both names for compatibility
                    sys.modules['keras.src.engine.functional'] = func_module
                    sys.modules[name] = func_module
                    return func_module
                except:
                    # Fallback: try keras.src.engine.functional (Keras 2.x / tf_keras)
                    try:
                        from keras.src.engine import functional as func_module
                        sys.modules[name] = func_module
                        return func_module
                    except:
                        # If neither works, try to get it from keras.src.engine directly
                        if hasattr(keras, 'src') and hasattr(keras.src, 'engine'):
                            if hasattr(keras.src.engine, 'functional'):
                                module = keras.src.engine.functional
                                sys.modules[name] = module
                                return module
                        # Last resort: create a fake module with Functional class
                        from types import ModuleType
                        fake_module = ModuleType('keras.src.engine.functional')
                        # Get Functional class - try multiple ways
                        try:
                            fake_module.Functional = keras.Model
                        except:
                            try:
                                import tensorflow as tf
                                fake_module.Functional = tf.keras.Model
                            except:
                                # Last resort: use a generic Model class
                                fake_module.Functional = type('Functional', (), {})
                        sys.modules[name] = fake_module
                        return fake_module
            except Exception as e:
                # If all else fails, create a minimal fake module
                from types import ModuleType
                fake_module = ModuleType('keras.src.engine.functional')
                try:
                    import keras
                    fake_module.Functional = keras.Model
                except:
                    try:
                        import tensorflow as tf
                        fake_module.Functional = tf.keras.Model
                    except:
                        pass
                sys.modules[name] = fake_module
                return fake_module
        
        # For all other imports, use original function
        return _original_import_module(name, package)
    
    # Patch importlib.import_module
    importlib.import_module = _patched_import_module
    
    # Also patch the Keras serialization library as backup
    try:
        from keras.src.saving import serialization_lib
        original_retrieve = serialization_lib._retrieve_class_or_fn
        original_deserialize = serialization_lib.deserialize_keras_object
        
        def patched_retrieve(name, registered_name, module, obj_type, full_config, custom_objects):
            # Map tf_keras module names to keras
            if module and module.startswith('tf_keras'):
                module = module.replace('tf_keras', 'keras', 1)
            return original_retrieve(name, registered_name, module, obj_type, full_config, custom_objects)
        
        def fix_keras2_to_keras3_compat(config_dict):
            """Recursively fix Keras 2.x -> 3.x compatibility issues"""
            if not isinstance(config_dict, dict):
                return config_dict
            
            config_dict = config_dict.copy()  # Don't modify the original
            
            # Fix this level based on layer type
            class_name = config_dict.get('class_name', '')
            inner_config = config_dict.get('config', {})
            
            if class_name == 'BatchNormalization' and isinstance(inner_config, dict):
                # Fix BatchNormalization axis: convert list to integer
                if 'axis' in inner_config and isinstance(inner_config['axis'], list):
                    inner_config = inner_config.copy()
                    inner_config['axis'] = inner_config['axis'][0] if inner_config['axis'] else -1
                    config_dict['config'] = inner_config
            
            elif class_name == 'LSTM' and isinstance(inner_config, dict):
                # Fix LSTM: remove time_major parameter (not in Keras 3.x)
                if 'time_major' in inner_config:
                    inner_config = inner_config.copy()
                    del inner_config['time_major']
                    config_dict['config'] = inner_config
            
            # Fix optimizer configs: remove jit_compile and is_legacy_optimizer (not in Keras 3.x)
            elif class_name in ('Adam', 'AdamW', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam'):
                if isinstance(inner_config, dict):
                    inner_config = inner_config.copy()
                    # Remove parameters not recognized in Keras 3.x
                    if 'jit_compile' in inner_config:
                        del inner_config['jit_compile']
                    if 'is_legacy_optimizer' in inner_config:
                        del inner_config['is_legacy_optimizer']
                    config_dict['config'] = inner_config
            
            # Also check compile_config for optimizer settings
            if 'compile_config' in config_dict and isinstance(config_dict['compile_config'], dict):
                compile_config = config_dict['compile_config'].copy()
                if 'optimizer' in compile_config and isinstance(compile_config['optimizer'], dict):
                    opt_config = compile_config['optimizer'].copy()
                    opt_class_name = opt_config.get('class_name', '')
                    opt_inner_config = opt_config.get('config', {})
                    if opt_class_name in ('Adam', 'AdamW', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam'):
                        if isinstance(opt_inner_config, dict):
                            opt_inner_config = opt_inner_config.copy()
                            if 'jit_compile' in opt_inner_config:
                                del opt_inner_config['jit_compile']
                            if 'is_legacy_optimizer' in opt_inner_config:
                                del opt_inner_config['is_legacy_optimizer']
                            opt_config['config'] = opt_inner_config
                            compile_config['optimizer'] = opt_config
                            config_dict['compile_config'] = compile_config
            
            # Recursively fix nested configs (e.g., in layers list)
            if 'layers' in config_dict and isinstance(config_dict['layers'], list):
                config_dict['layers'] = [fix_keras2_to_keras3_compat(layer) for layer in config_dict['layers']]
            elif 'config' in config_dict and isinstance(config_dict['config'], dict):
                # Recursively fix nested config
                config_dict['config'] = fix_keras2_to_keras3_compat(config_dict['config'])
            
            return config_dict
        
        def patched_deserialize(config, custom_objects=None, safe_mode=False, **kwargs):
            """Patched deserialize to fix Keras 2.x -> 3.x compatibility issues"""
            # Fix Keras 2.x -> 3.x compatibility issues (BatchNormalization axis, LSTM time_major, etc.)
            if isinstance(config, dict):
                config = fix_keras2_to_keras3_compat(config)
            
            return original_deserialize(config, custom_objects=custom_objects, safe_mode=safe_mode, **kwargs)
        
        serialization_lib._retrieve_class_or_fn = patched_retrieve
        serialization_lib.deserialize_keras_object = patched_deserialize
        
        # Also patch BatchNormalization.from_config directly to handle axis conversion
        try:
            from keras.src.layers.normalization.batch_normalization import BatchNormalization
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
        except (ImportError, AttributeError):
            pass
        
        # Patch LSTM.from_config to remove time_major parameter (not in Keras 3.x)
        try:
            from keras.src.layers.rnn.lstm import LSTM
            original_lstm_from_config = LSTM.from_config
            
            @classmethod
            def patched_lstm_from_config(cls, config):
                """Patched from_config to remove time_major parameter (Keras 2.x -> 3.x)"""
                if isinstance(config, dict):
                    config = config.copy()
                    # Remove time_major if present (not supported in Keras 3.x)
                    if 'time_major' in config:
                        del config['time_major']
                return original_lstm_from_config(config)
            
            LSTM.from_config = patched_lstm_from_config
        except (ImportError, AttributeError):
            pass
        
        # Patch optimizer from_config methods to remove jit_compile and is_legacy_optimizer
        def create_optimizer_patcher(original_from_config):
            @classmethod
            def patched_optimizer_from_config(cls, config):
                """Patched from_config to remove jit_compile and is_legacy_optimizer (Keras 2.x -> 3.x)"""
                if isinstance(config, dict):
                    config = config.copy()
                    # Remove parameters not recognized in Keras 3.x
                    if 'jit_compile' in config:
                        del config['jit_compile']
                    if 'is_legacy_optimizer' in config:
                        del config['is_legacy_optimizer']
                return original_from_config(config)
            return patched_optimizer_from_config
        
        # Patch common optimizers
        optimizer_classes = ['Adam', 'AdamW', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']
        for opt_name in optimizer_classes:
            try:
                from keras.src.optimizers import optimizers
                if hasattr(optimizers, opt_name):
                    opt_class = getattr(optimizers, opt_name)
                    if hasattr(opt_class, 'from_config'):
                        original_opt_from_config = opt_class.from_config
                        opt_class.from_config = create_optimizer_patcher(original_opt_from_config)
            except (ImportError, AttributeError):
                # Try alternative import path
                try:
                    opt_module = __import__(f'keras.src.optimizers.{opt_name.lower()}', fromlist=[opt_name])
                    opt_class = getattr(opt_module, opt_name)
                    if hasattr(opt_class, 'from_config'):
                        original_opt_from_config = opt_class.from_config
                        opt_class.from_config = create_optimizer_patcher(original_opt_from_config)
                except (ImportError, AttributeError):
                    pass
    except (ImportError, AttributeError) as e:
        pass
    
    # Pre-register key modules - ensure they're registered as packages
    sys.modules['tf_keras'] = keras
    if hasattr(keras, 'src'):
        sys.modules['tf_keras.src'] = keras.src
        # Ensure it has __path__ to be recognized as a package
        if not hasattr(sys.modules['tf_keras.src'], '__path__') and hasattr(keras.src, '__path__'):
            sys.modules['tf_keras.src'].__path__ = keras.src.__path__
        
        if hasattr(keras.src, 'engine'):
            sys.modules['tf_keras.src.engine'] = keras.src.engine
            # Ensure it has __path__ to be recognized as a package
            if not hasattr(sys.modules['tf_keras.src.engine'], '__path__') and hasattr(keras.src.engine, '__path__'):
                sys.modules['tf_keras.src.engine'].__path__ = keras.src.engine.__path__
            
            # Try to import and register functional module
            try:
                functional_module = _original_import_module('keras.src.engine.functional')
                sys.modules['tf_keras.src.engine.functional'] = functional_module
            except Exception as e:
                # If direct import fails, try using the patched import_module
                try:
                    functional_module = importlib.import_module('tf_keras.src.engine.functional')
                except:
                    pass

print("✓ Keras 2.x to 3.x compatibility patches loaded")

def plot_loss_curves(model_dir, BL_abbr, show_plot=False):
    """
    Plot and save training and validation loss curves from a saved training history file.
    Args:
        model_dir (str): Directory where the training history file is stored.
        BL_abbr (str): Abbreviation for the model/baseline, used in filename.
        show_plot (bool): If True, display the plot interactively.
    """
    final_loss = []
    final_val_loss = []
    nan = np.nan  # for possible use in eval

    # Read loss values from the training history file
    with open(os.path.join(model_dir, f'traininghistory_CNN_{BL_abbr}.txt')) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # The next line after the marker contains the loss list
            if 'final_loss_median_model' in line:
                final_loss = eval(lines[i + 1])
            if 'final_val_loss_median_model' in line:
                final_val_loss = eval(lines[i + 1])

    fontsize = 16

    # Plot training and validation loss
    try:
        plt.plot(final_loss, '#244062')      # Training loss in blue
        plt.plot(final_val_loss, '#EA3355')  # Validation loss in red
        plt.ylabel('MSE loss (m asl)', fontsize=fontsize)
        plt.xlabel('epoch', fontsize=fontsize)
        plt.tick_params(axis='both', labelsize=fontsize)
        plt.legend(['training loss', 'validation loss'], loc='upper right', fontsize=fontsize)
        plt.savefig(model_dir + '/' + 'loss_curves_.png', dpi=300)
        if show_plot:
            plt.show()
        plt.close()
    except Exception as e:
        print(f"[WARNING] Plotting error in plot_loss_curves: {e}")


# def plot_r2_rmse_boxplots(df, model_dir, fontsize=16, show_plot=False):
#     """
#     Plots boxplots for R2 and RMSE from a scores DataFrame and saves the figure.
#     Args:
#         df (pd.DataFrame): DataFrame with 'R2' and 'RMSE' columns.
#         model_dir (str): Directory to save the plot.
#         fontsize (int): Font size for plot labels.
#         show_plot (bool): Whether to display the plot with plt.show().
#     """
#     fig, ax = plt.subplots(1, 2, figsize=(8, 5))

#     # R2 boxplot
#     sns.boxplot(
#         y=df['R2'], width=0.5, color="#FFD7DF",
#         ax=ax[0],
#         boxprops=dict(edgecolor="black", linewidth=1.5),
#         whiskerprops=dict(color="black", linewidth=1.5),
#         capprops=dict(color="black", linewidth=1.5),
#         medianprops=dict(color="#244062", linewidth=2)
#     )
#     # Show R2 summary statistics on the plot
#     s = f"R2 \nmedian = {round(df['R2'].median(),2)}\nmax = {round(df['R2'].max(),2)}\nmin = {round(df['R2'].min(),2)}"
#     ax[0].text(0.6, -0.025, s, bbox=dict(facecolor='white'), fontsize=fontsize)
#     ax[0].set_ylabel("R2", fontsize=fontsize)
#     ax[0].tick_params(axis='both', labelsize=fontsize-2)

#     # RMSE boxplot
#     sns.boxplot(
#         y=df['RMSE'], width=0.5, color="#FFD7DF",
#         ax=ax[1],
#         boxprops=dict(edgecolor="black", linewidth=1.5),
#         whiskerprops=dict(color="black", linewidth=1.5),
#         capprops=dict(color="black", linewidth=1.5),
#         medianprops=dict(color="#244062", linewidth=2)
#     )
#     # Show RMSE summary statistics on the plot
#     s = f"RMSE \nmedian = {round(df['RMSE'].median(),2)}\nmax = {round(df['RMSE'].max(),2)}\nmin = {round(df['RMSE'].min(),2)}"
#     ax[1].text(0.6, 0.035, s, bbox=dict(facecolor='white'), fontsize=fontsize)
#     ax[1].set_ylabel("RMSE", fontsize=fontsize)
#     ax[1].tick_params(axis='both', labelsize=fontsize-2)

#     plt.tight_layout()
#     plt.savefig(f"{model_dir}/boxplot_r2_rmse.png", dpi=300)
#     if show_plot:
#         plt.show()
#     plt.close()


def plot_simulation_vs_observed(TestData, scores, sim, obs, sim_mean_uncertainty, sim_members, WELL_ID, inimax, GLOBAL_SETTINGS, BL_abbr, save_fig=False):
    """
    Plots the observed vs. simulated data with a 95% confidence interval.
     inspired and adapted from Wunsch et al 2022 on github https://github.com/AndreasWunsch/Long-Term-GWL-Simulations

    Parameters:
    - TestData: DataFrame with the index as the time series (e.g., dates).
    - sim: Numpy array of simulated median values.
    - obs: Numpy array of observed values.
    - sim_mean_uncertainty: Array of uncertainties associated with the simulated data.
    - WELL_ID: number of well IDs for labeling the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from uncertainties import unumpy

    try:
        # Calculate the standard deviations of uncertainties
        y_err = unumpy.std_devs(sim_mean_uncertainty)

        plt.figure(figsize=(20, 6))
        
        for i in range(inimax):
            sim_runs = np.asarray(sim_members[:,i].reshape(-1,1))
            plt.plot(TestData.index, sim_runs, linewidth=1, color="grey")
        plt.plot(TestData.index, sim_runs, linewidth=1, color="grey", label="Simulation runs")
        plt.plot(TestData.index, sim, 'r', label="Simulated Median", linewidth=1.7)
        
        # Plot the observed data
        plt.plot(TestData.index, obs, 'k', label="Observed", linewidth=1.7, alpha=0.9)
        
        # Titles and labels
        plt.title("CNN Model Test: " + BL_abbr + WELL_ID, size=17, fontweight='bold')
        plt.ylabel('GWL [m asl]', size=15)
        plt.xlabel('Date', size=15)
        
        s = """NSE = {:.2f}\nR²  = {:.2f}\nRMSE = {:.2f}\nrRMSE = {:.2f}
Bias = {:.2f}\nrBias = {:.2f}\n\nfilters = {:d}\ndense-size = {:d}\nseqlength = {:d}
batchsize = {:d}\n""".format(
            scores.NSE[0], scores.R2[0], scores.RMSE[0], scores.rRMSE[0], scores.Bias[0], scores.rBias[0],
            GLOBAL_SETTINGS["filters"], GLOBAL_SETTINGS["dense_size"], GLOBAL_SETTINGS["window_size"], GLOBAL_SETTINGS["batch_size"]
        )
        plt.figtext(0.872, 0.18, s, bbox=dict(facecolor='white'), fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1.18, 1), loc='upper right', fancybox=False, framealpha=1, edgecolor='k')
        plt.tight_layout()
        if save_fig:
            plt.savefig('./Test_'+WELL_ID+'_CNN.png', dpi=300)            
        plt.close()
    except Exception as e:
        print(f"[WARNING] Plotting error in plot_simulation_vs_observed: {e}")


###################
## Function Wrapper: Confidence interval instead of singualr inititations
def simulate_plot_wells_testset_conf_int(well_ids, TestData_dict, GLOBAL_SETTINGS, model_path, inimax, number_of_wells, columns_to_keep=None, BL_abbr = "BB", save_fig = False):
    """ 
    Analyzes test data for a specified number of wells using a trained CNN model.
     inspired and adapted from Wunsch et al 2022 on github https://github.com/AndreasWunsch/Long-Term-GWL-Simulations

    Parameters:
        well_ids (list): List of well IDs.
        TestData_dict (dict): Dictionary containing test data and observation dataframes.
        GLOBAL_SETTINGS (dict): Global settings for the analysis.
        model_path (str): Path to the trained model weights.
        inimax (int): Number of simulation runs for uncertainty analysis.
        number_of_wells (int): Number of wells to analyze.
        columns_to_keep (list): List of feature names.
        BL_abbr (str): Abbreviation for the region/model.
        save_fig (bool): Whether to save figures.

    Returns:
        dict: Dictionary containing scores for each well.
    """
    # If columns_to_keep is not provided, try to extract it from TestData_dict
    if columns_to_keep is None:
        # Try to get columns from the first well's dataframe
        first_well_id = well_ids[0] if well_ids else None
        if first_well_id and f'obs_Dataframe_{first_well_id}' in TestData_dict:
            data_cols = list(TestData_dict[f'obs_Dataframe_{first_well_id}'].columns)
            # Remove 'GWL' and add it back at the end if present
            if 'GWL' in data_cols:
                data_cols.remove('GWL')
                columns_to_keep = data_cols + ['GWL']
            else:
                columns_to_keep = data_cols
        else:
            # Fallback: use a default value based on X_test shape
            if first_well_id and f'X_test_{first_well_id}' in TestData_dict:
                num_features = TestData_dict[f'X_test_{first_well_id}'].shape[2]
                columns_to_keep = [f'feature_{i}' for i in range(num_features)] + ['GWL']
            else:
                columns_to_keep = ['GWL']  # Minimal fallback
        print(f"[INFO] columns_to_keep not provided, inferred from data: {len(columns_to_keep)} features")
    
    # Apply BatchNormalization axis fix patch before loading any models
    # This fixes the issue where axis is saved as a list [2] but Keras 3.x expects an integer
    try:
        from keras.src.layers.normalization.batch_normalization import BatchNormalization
        from keras.src.saving import serialization_lib
        
        # Patch BatchNormalization.from_config to handle axis conversion
        if not hasattr(BatchNormalization, '_patched_for_axis_fix_in_simulate'):
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
            BatchNormalization._patched_for_axis_fix_in_simulate = True
            
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
            print("✓ Applied BatchNormalization axis fix patch in simulate_plot_wells_testset_conf_int")
    except Exception as e:
        print(f"[WARNING] Could not apply BatchNormalization patch in simulate_plot_wells_testset_conf_int: {e}")
    
    scores_dict = {}

    for number_of_well in range(number_of_wells):
        WELL_ID = well_ids[number_of_well]
        

        # Extract test data for the current well
        X_test = TestData_dict[f'X_test_{WELL_ID}']
        Y_test = TestData_dict[f'Y_test_{WELL_ID}']
        data = TestData_dict[f'obs_Dataframe_{WELL_ID}'].copy()
        TestData_cut = TestData_dict[f'obs_Dataframe_{WELL_ID}'].iloc[GLOBAL_SETTINGS["window_size"]:]

        # Initialize simulation arrays
        sim_members = np.zeros((len(X_test), inimax))
        sim_members[:] = np.nan
        sim_std = np.zeros((len(X_test), inimax))
        sim_std[:] = np.nan

        ## initialise individual scaler again
        _, scaler_y, _ = scale_dataset_indiv(data, target_column = "GWL", )



        # Run simulations for each initial condition
        for ini in range(inimax):
            # Load model with device placement handling for GPU-trained models on CPU
            # Use compile=False to avoid compilation issues and force CPU placement
            try:
                import keras
                # Get the Functional class from keras
                Functional = keras.src.engine.functional.Functional
            except:
                try:
                    Functional = keras.Model
                except:
                    Functional = tf.keras.Model
            
            custom_objects = {'Functional': Functional}
            
            loaded_model = None
            try:
                with tf.device('/CPU:0'):
                    loaded_model = tf.keras.models.load_model(
                        model_path + f'model_weights_{BL_abbr}_run_{ini}.keras',
                        compile=False,
                        custom_objects=custom_objects
                    )
            except RecursionError as e:
                # Recursion error - try with increased recursion limit
                import sys
                original_limit = sys.getrecursionlimit()
                try:
                    sys.setrecursionlimit(original_limit * 2)
                    print(f"Warning: Recursion error during model load, temporarily increasing recursion limit to {sys.getrecursionlimit()}")
                    try:
                        with tf.device('/CPU:0'):
                            loaded_model = tf.keras.models.load_model(
                                model_path + f'model_weights_{BL_abbr}_run_{ini}.keras',
                                compile=False,
                                custom_objects=custom_objects
                            )
                    except Exception as e2:
                        print(f"Warning: Still failed with increased recursion limit: {e2}")
                        print(f"Attempting to load model without explicit device placement...")
                        loaded_model = tf.keras.models.load_model(
                            model_path + f'model_weights_{BL_abbr}_run_{ini}.keras',
                            compile=False,
                            custom_objects=custom_objects
                        )
                finally:
                    sys.setrecursionlimit(original_limit)
            except Exception as e:
                # If loading fails, try without device context (fallback)
                print(f"Warning: Failed to load model with CPU device context: {e}")
                print(f"Attempting to load model without explicit device placement...")
                try:
                    loaded_model = tf.keras.models.load_model(
                        model_path + f'model_weights_{BL_abbr}_run_{ini}.keras',
                        compile=False,
                        custom_objects=custom_objects
                    )
                except RecursionError as e2:
                    # Final fallback: try with increased recursion limit
                    import sys
                    original_limit = sys.getrecursionlimit()
                    try:
                        sys.setrecursionlimit(original_limit * 2)
                        print(f"Warning: Recursion error, temporarily increasing recursion limit to {sys.getrecursionlimit()}")
                        loaded_model = tf.keras.models.load_model(
                            model_path + f'model_weights_{BL_abbr}_run_{ini}.keras',
                            compile=False,
                            custom_objects=custom_objects
                        )
                    finally:
                        sys.setrecursionlimit(original_limit)
            
            if loaded_model is None:
                raise RuntimeError(f"Failed to load model for run {ini} after all retry attempts")
            
            # Fix model attributes that might be None when loading with compile=False
            # This is needed for Keras 3.x compatibility when loading models saved with Keras 2.x
            if hasattr(loaded_model, 'steps_per_execution') and loaded_model.steps_per_execution is None:
                loaded_model.steps_per_execution = 1
            if hasattr(loaded_model, 'jit_compile') and loaded_model.jit_compile is None:
                loaded_model.jit_compile = False
            if hasattr(loaded_model, 'run_eagerly') and loaded_model.run_eagerly is None:
                loaded_model.run_eagerly = False
            
            y_pred_distribution = predict_distribution(X_test, loaded_model, 100)
            sim = scaler_y.inverse_transform(y_pred_distribution)
            sim_members[:, ini], sim_std[:, ini] = sim.mean(axis=1), sim.std(axis=1)

        # Calculate uncertainties and statistics
        sim_members_uncertainty = unumpy.uarray(sim_members, 1.96 * sim_std)
        sim_mean = np.nanmedian(sim_members, axis=1)
        sim_mean_uncertainty = np.sum(sim_members_uncertainty, axis=1) / inimax

        sim_max =  np.nanmax(sim_members,axis = 1)
        sim_min =  np.nanmin(sim_members,axis = 1)

        sim = np.asarray(sim_mean.reshape(-1, 1))
        obs = np.asarray(scaler_y.inverse_transform(Y_test.reshape(-1, 1)))

        err = sim - obs
        err_rel = (sim - obs) / (np.max(data['GWL']) - np.min(data['GWL']))
        err_nash = obs - np.mean(np.asarray(data['GWL'][(data.index < GLOBAL_SETTINGS["test_start"])]))

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
            columns=['R2', "NSE",'RMSE', 'rRMSE', 'Bias', 'rBias']
        )
        scores_dict[WELL_ID] = scores

        # Plot results
        plt.figure(figsize=(20, 6))
        fontsize = 17
       # for ini in range(inimax):
       #     sim_runs = np.asarray(sim_members[:, ini].reshape(-1, 1))
       #     plt.plot(TestData_cut.index, sim_members[:, ini], linewidth=1, color="grey")

        try:
            plt.fill_between(TestData_cut.index, 
                 sim_max,sim_min, facecolor = "#C8DAFB", alpha = 0.7,
                label ='Uncertainty',linewidth = 1,
                    #edgecolor = (1,0.7,0,0.7)
                )

      #  plt.plot(TestData_cut.index, sim_runs, linewidth=1, color="grey", label="Simulation runs")
            plt.plot(TestData_cut.index, sim, '#EA3358', label="Simulated Median", linewidth=1.7)
            plt.plot(TestData_cut.index, obs, 'k', label="Observed Data", linewidth=1.7, alpha=0.9)
            plt.title(f"CNN Model Run: {BL_abbr}{WELL_ID}", size=fontsize+2, fontweight='bold')
            plt.ylabel('GWL [m asl]', size=fontsize)
            plt.xlabel('Date', size=fontsize)

            # Add text box with statistics, now including NSE
            s1 = """R² = {:.2f}\nNSE = {:.2f}\nRMSE = {:.2f}\nrRMSE = {:.2f}\nBias = {:.2f}\nrBias = {:.2f}""".format(
                scores.R2[0], 
                scores.NSE[0],
                scores.RMSE[0], 
                scores.rRMSE[0], 
                scores.Bias[0], 
                scores.rBias[0],)


            # Add text box with hyperparameters
            s2 = """number of input parameters = {:d}\nfilters = {:d}\ndense-size = {:d}\nwindow size = {:d}\nbatchsize = {:d}""".format(

            len(columns_to_keep)-1, 
            GLOBAL_SETTINGS["filters"], 
            GLOBAL_SETTINGS["dense_size"], 
            GLOBAL_SETTINGS["window_size"],
            GLOBAL_SETTINGS["batch_size"])
            
            plt.figtext(0.849, 0.45, s1, bbox=dict(facecolor='white'), fontsize=fontsize)
           # plt.figtext(0.872, 0.14, s2, bbox=dict(facecolor='white'), fontsize=fontsize)
            plt.legend(fontsize=fontsize, bbox_to_anchor=(1.21, 1), loc='upper right', fancybox=False, framealpha=1, edgecolor='k')
              # ticks
            plt.tick_params(axis='both', labelsize=fontsize-2)  # Change font size of both x and y ticks

            plt.grid()
            plt.tight_layout()
            

            if save_fig == True:
                from_year, to_year = TestData_cut.index.min().year, TestData_cut.index.max().year
                plt.savefig(model_path+BL_abbr +'_'+ WELL_ID+'_CNN_run_'+str(from_year)+'_'+str(to_year)+'.png', dpi=300)   
            plt.close() 
          #   plt.show()
        except Exception as e:
            print(f"[WARNING] Plotting error in simulate_plot_wells_testset_conf_int: {e}")

    return scores_dict


def feature_importance(median_model, X_test, Y_test, scaler_y, columns_to_keep, output_dir, output_filename):
    """
    Computes feature importance by shuffling each feature and measuring performance drop.
    Saves results to a .txt file with columns: feature_name, R2, RMSE.

    Args:
        median_model: Trained model for prediction.
        X_test: Test set input data (3D array).
        Y_test: Test set target data.
        scaler_y: Scaler used to inverse-transform predictions and targets.
        columns_to_keep: List of feature names (order must match X_test).
        output_dir: Directory to save the results file.
        output_filename: Name of the output .txt file.
    """
    import os
    obs = np.asarray(scaler_y.inverse_transform(Y_test.reshape(-1,1)))  # True values, inverse scaled
    results = []
    for feature in range(X_test.shape[2]):
        # Shuffle one feature at a time to break its relationship with the target
        X_modified = X_test.copy()
        random.shuffle(X_modified[:,:,feature])  # In-place shuffle along the sample axis
        y_predictions = median_model.predict(X_modified)
        sim = scaler_y.inverse_transform(y_predictions)  # Inverse scale predictions
        err = sim - obs
        r = stats.linregress(sim[:,0], obs[:,0])
        R2 = r.rvalue ** 2
        RMSE = np.sqrt(np.mean(err ** 2))
        # If 'GWL' is the first column, skip it in feature names
        feature_name = columns_to_keep[feature+1] if columns_to_keep[0] == 'GWL' else columns_to_keep[feature]
        print(f"Performance with {feature_name} removed: (R2={R2}, RMSE={RMSE})")
        results.append((feature_name, R2, RMSE))
    # Save results to file (tab-separated)
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        f.write('feature_name\tR2\tRMSE\n')
        for feature_name, R2, RMSE in results:
            f.write(f'{feature_name}\t{R2}\t{RMSE}\n')


# # Example usage for all wells:
# # Concatenate test data from all wells for global feature importance analysis
# X_test_all = np.concatenate([TestData_dict[f'X_test_{well_id}'] for well_id in well_ids], axis=0)
# Y_test_all = np.concatenate([TestData_dict[f'Y_test_{well_id}'] for well_id in well_ids], axis=0)
#
# feature_importance(
#     median_model,
#     X_test_all,
#     Y_test_all,
#     scaler_y,
#     columns_to_keep,
#     model_path,         # output_dir
#     "feature_importance.txt"         # output_filename
# )



def compute_and_save_shap_values(median_model, X_train, X_test_all, model_dir, columns_to_keep, nsamples=100, hide_logging=True):
    """
    Computes SHAP values for the given model and test set, and saves them along with the input data to a .txt file.
    Args:
        median_model: Trained Keras model.
        X_train: Training data (background for SHAP).
        X_test: Test data to explain.
        model_dir: Directory to save the output file.
        BL_abbr: String for file naming.
        columns_to_keep: List of feature names.
    """
    import shap
    import numpy as np
    import os
    import logging

    # Hide verbose output from SHAP KernelExplainer only if hide_logging is True
    if hide_logging:
        logging.getLogger("shap").setLevel(logging.ERROR)
        logging.getLogger("shap.common").setLevel(logging.ERROR)
        logging.getLogger("shap.explainers").setLevel(logging.ERROR)
        logging.getLogger("shap.explainers._kernel").setLevel(logging.ERROR)

    # Compute SHAP values
    # background = X_train
    # # Use a smaller background set
    #background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
    background = X_train[np.random.choice(X_train.shape[0], min(nsamples, X_train.shape[0]), replace=False)]
    X_test_last = X_test_all[:, -1, :]  # shape: (samples, features)
    background_last = background[:, -1, :]

    explainer = shap.KernelExplainer(lambda x: median_model.predict(x.reshape((x.shape[0], 1, x.shape[1]))), background_last)
    print("Explainer created")
    shap_values = explainer.shap_values(X_test_last, nsamples=nsamples)
    #shap_values = e.shap_values(X_test)
    # Reshape for saving
    shap_vals = shap_values[:,:,0]# remove last dimension
    #shap_vals = np.asarray(shap_values[:,:,:,0]) # remove last dimension
    #shap_vals = shap_vals.reshape(-1, shap_vals.shape[-1])
    #x = X_test.reshape(-1, X_test.shape[-1])
    # Save to file
    output_path = os.path.join(model_dir, f"shapvalues.txt")
    with open(output_path, "w") as f:
        f.write('shap_vals\n')
        for row in shap_vals:
            f.write(' '.join(map(str, row)) + '\n')
        # The variable 'X_test_last' is intended to be the input data corresponding to the SHAP values, 
        f.write('input_data\n')
        for row in X_test_last:
            f.write(' '.join(map(str, row)) + '\n')
    
    return shap_vals, X_test_last


import numpy as np
import shap
import os
import logging
from sklearn.cluster import KMeans

def save_shap_to_file(shap_vals, input_data, output_path):
    """Save SHAP values and input data to a text file"""
    with open(output_path, "w") as f:
        f.write('shap_vals\n')
        for row in shap_vals:
            f.write(' '.join(map(str, row)) + '\n')
        f.write('input_data\n')
        for row in input_data:
            f.write(' '.join(map(str, row)) + '\n')



def compute_and_save_shap_values_robust(median_model, X_train, X_test_all, model_dir, columns_to_keep, 
                                       nsamples=100, background_size=100, use_kmeans_background=True, 
                                       hide_logging=True, stability_check=True):
    """
    Computes SHAP values for the given model and test set with improved numerical stability.
    
    Args:
        median_model: Trained Keras model.
        X_train: Training data (background for SHAP).
        X_test_all: Test data to explain.
        model_dir: Directory to save the output file.
        columns_to_keep: List of feature names.
        nsamples: Number of samples for SHAP computation.
        background_size: Size of background dataset.
        use_kmeans_background: Whether to use k-means clustering for background selection.
        hide_logging: Whether to hide verbose SHAP logging.
        stability_check: Whether to perform stability checks on SHAP values.
    """
    
    # Hide verbose output from SHAP KernelExplainer only if hide_logging is True
    if hide_logging:
        logging.getLogger("shap").setLevel(logging.ERROR)
        logging.getLogger("shap.common").setLevel(logging.ERROR)
        logging.getLogger("shap.explainers").setLevel(logging.ERROR)
        logging.getLogger("shap.explainers._kernel").setLevel(logging.ERROR)

    print(f"Computing SHAP values for {X_test_all.shape[0]} samples with {X_test_all.shape[2]} features")
    
    # Extract last time step for SHAP analysis
    X_test_last = X_test_all[:, -1, :]  # shape: (samples, features)
    X_train_last = X_train[:, -1, :]    # shape: (samples, features)
    
    # Create a more representative background dataset
    if use_kmeans_background and X_train_last.shape[0] > background_size:
        print(f"Using k-means clustering to select {background_size} representative background samples")
        kmeans = KMeans(n_clusters=background_size, random_state=42, n_init=10)
        kmeans.fit(X_train_last)
        background_last = kmeans.cluster_centers_
    else:
        # Use random sampling for background
        background_indices = np.random.choice(X_train_last.shape[0], 
                                            min(background_size, X_train_last.shape[0]), 
                                            replace=False)
        background_last = X_train_last[background_indices]
    
    print(f"Background dataset shape: {background_last.shape}")
    
    # Create a more stable prediction function
    def stable_predict(x):
        """Wrapper for model prediction with numerical stability checks"""
        try:
            # Reshape input for LSTM model
            x_reshaped = x.reshape((x.shape[0], 1, x.shape[1]))
            
            # Clip input to reasonable range to prevent numerical issues
            x_clipped = np.clip(x_reshaped, -10, 10)
            
            # Make prediction
            pred = median_model.predict(x_clipped, verbose=0)
            
            # Check for numerical issues in predictions
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                print(f"WARNING: Found NaN/inf in predictions, replacing with zeros")
                pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            
            return pred
            
        except Exception as e:
            print(f"ERROR in prediction function: {e}")
            # Return zeros as fallback
            return np.zeros((x.shape[0], 1))
    
    # Test the prediction function with background data
    print("Testing prediction function stability...")
    test_pred = stable_predict(background_last[:5])
    print(f"Test prediction shape: {test_pred.shape}, range: [{test_pred.min():.6f}, {test_pred.max():.6f}]")
    
    # Create SHAP explainer with improved settings
    print("Creating SHAP explainer...")
    explainer = shap.KernelExplainer(stable_predict, background_last)
    
    # Compute SHAP values with smaller batches to avoid memory/numerical issues
    batch_size = min(1000, X_test_last.shape[0])  # Process in batches
    all_shap_values = []
    
    print(f"Computing SHAP values in batches of {batch_size}...")
    for i in range(0, X_test_last.shape[0], batch_size):
        end_idx = min(i + batch_size, X_test_last.shape[0])
        batch_data = X_test_last[i:end_idx]
        
        print(f"Processing batch {i//batch_size + 1}/{(X_test_last.shape[0] + batch_size - 1)//batch_size}")
        
        try:
            # Compute SHAP values for this batch
            batch_shap = explainer.shap_values(batch_data, nsamples=nsamples, silent=True)
            
            # Handle different SHAP output formats
            if isinstance(batch_shap, list):
                batch_shap = batch_shap[0]  # Take first output for regression
            
            # Remove extra dimensions if present
            if len(batch_shap.shape) > 2:
                batch_shap = batch_shap[:, :, 0] if batch_shap.shape[2] == 1 else batch_shap.reshape(batch_shap.shape[0], -1)
            
            all_shap_values.append(batch_shap)
            
        except Exception as e:
            print(f"ERROR computing SHAP values for batch {i//batch_size + 1}: {e}")
            # Create zero SHAP values as fallback
            fallback_shap = np.zeros((batch_data.shape[0], batch_data.shape[1]))
            all_shap_values.append(fallback_shap)
    
    # Concatenate all batches
    shap_vals = np.concatenate(all_shap_values, axis=0)
    
    print(f"Final SHAP values shape: {shap_vals.shape}")
    print(f"SHAP values range: [{shap_vals.min():.2e}, {shap_vals.max():.2e}]")
    
    # Stability check
    if stability_check:
        extreme_mask = np.abs(shap_vals) > 1.0  # Flag values > 1.0 as potentially problematic
        n_extreme = np.sum(extreme_mask)
        
        if n_extreme > 0:
            print(f"WARNING: Found {n_extreme} potentially extreme SHAP values (>1.0)")
            
            # Option to clip extreme values
            shap_vals_clipped = np.clip(shap_vals, -1.0, 1.0)
            
            # Save both original and clipped versions
            output_path_original = os.path.join(model_dir, "shapvalues_original.txt")
            output_path_clipped = os.path.join(model_dir, "shapvalues_robust.txt")
            
            # Save original
            save_shap_to_file(shap_vals, X_test_last, output_path_original)
            print(f"Original SHAP values saved to: {output_path_original}")
            
            # Save clipped version
            save_shap_to_file(shap_vals_clipped, X_test_last, output_path_clipped)
            print(f"Robust (clipped) SHAP values saved to: {output_path_clipped}")
            
            return shap_vals_clipped, X_test_last
        else:
            print("SHAP values appear stable (no extreme values found)")
    
    # Save to file
    output_path = os.path.join(model_dir, "shapvalues_improved.txt")
    save_shap_to_file(shap_vals, X_test_last, output_path)
    print(f"SHAP values saved to: {output_path}")
    
    return shap_vals, X_test_last



# example usage
# shap_vals, X_test_last = compute_and_save_shap_values(
#     median_model=median_model,
#     X_train=X_train,
#     X_test_all=X_test_all,
#     model_dir=model_path,
#     nsamples=10, #100 # You can omit this to use the default
#     columns_to_keep=columns_to_keep,
#     hide_logging=True

# )


#simple code that workd: 
# # Use a smaller background set
# background = X_train[np.random.choice(X_train.shape[0], min(100, X_train.shape[0]), replace=False)]
# X_test_last = X_test_all[:, -1, :]  # shape: (samples, features)
# background_last = background[:, -1, :]

# explainer = shap.KernelExplainer(lambda x: median_model.predict(x.reshape((x.shape[0], 1, x.shape[1]))), background_last)
# shap_values = explainer.shap_values(X_test_last, nsamples=100)
# plt.figure(figsize=(15, 5))


# shap.summary_plot(shap_values[:,:,0], X_test_last, feature_names=columns_to_keep, show=False)

# # plt.title(f"SHAP Values for CNN, all wells")
# plt.xlabel("SHAP value", fontsize=15)

# # plt.savefig(f"{model_dir}/"+"shap_values_"+str(len(columns_to_keep)-1)+'_params.png', dpi=300)            
# #plt.close()   
# plt.show()

def plot_shap_from_txt(txt_path, columns_to_keep, model_dir, custom_labels=None, show_plot=True):
    """
    Loads SHAP values and input data from a .txt file and plots a SHAP summary plot.
    Args:
        txt_path: Path to the .txt file containing SHAP values and input data.
        columns_to_keep: List of feature names.
        model_dir: Directory where the plot will be saved.
        custom_labels: List of custom y-axis labels for the SHAP plot (optional).
        show_plot: Whether to display the plot (default: True).
    """
    import shap  # For SHAP values

    with open(txt_path, 'r') as f:
        lines = f.readlines()
    # Find the split between shap_vals and input_data
    shap_start = lines.index('shap_vals\n') + 1
    input_start = lines.index('input_data\n') + 1
    shap_lines = lines[shap_start:input_start-1]
    input_lines = lines[input_start:]
    # Convert to numpy arrays
    shap_vals = np.array([list(map(float, line.strip().split())) for line in shap_lines])
    input_data = np.array([list(map(float, line.strip().split())) for line in input_lines])
    # Plot
    try:
        feature_names = custom_labels if custom_labels is not None else columns_to_keep
        shap.summary_plot(shap_vals, input_data, feature_names=feature_names, show=False)
        plt.xlabel("SHAP value", fontsize=15)
        # Make horizontal grid lines dotted and linewidth 1
        ax = plt.gca()
        ax.yaxis.grid(True, linestyle=':', linewidth=1)
        plt.savefig(f"{model_dir}/shap_values_plot.png", dpi=300)
        if show_plot:
            plt.show()
        plt.close()
    except Exception as e:
        print(f"[WARNING] Plotting error in plot_shap_from_txt: {e}")


def plot_shap_by_landcover(txt_path, columns_to_keep, model_dir, custom_labels=None, show_plot=True,
                          clc_legend_path="/Users/marie-christineckert/Nextcloud/TU/phd_work/GW_preprocessing_BB_all/meta_data/CORINE_LandCover/clc_legend.csv",
                          lookup_table_path="/Users/marie-christineckert/Nextcloud/TU/phd_work/GW_preprocessing_BB_all/data/GWData_BB_raw/BB_GW_wells_metadata_coords_4_visual_anthropo_217.csv"):
    """
    Loads SHAP values grouped by landcover from a .txt file and plots a SHAP summary plot for each landcover class.
    Each landcover gets its own plot window.
    
    Args:
        txt_path: Path to the .txt file containing SHAP values grouped by landcover.
        columns_to_keep: List of feature names.
        model_dir: Directory where the plots will be saved.
        custom_labels: List of custom y-axis labels for the SHAP plot (optional).
        show_plot: Whether to display the plots (default: True).
        clc_legend_path: Path to the CLC legend CSV file for decoding land cover names.
        lookup_table_path: Path to the metadata CSV file to count number of wells per land cover class.
    """
    import shap  # For SHAP values
    import pandas as pd
    
    # Load CLC legend for decoding land cover names
    try:
        clc_legend_df = pd.read_csv(clc_legend_path)
        # Create mapping from CLC_CODE to LABEL2
        clc_legend_df['CLC_CODE_STR'] = clc_legend_df['CLC_CODE'].astype(str).str.replace('.0', '', regex=False)
        label2_mapping = dict(zip(clc_legend_df['CLC_CODE_STR'], clc_legend_df['LABEL2']))
        label3_mapping = dict(zip(clc_legend_df['CLC_CODE_STR'], clc_legend_df['LABEL3']))
        print(f"Loaded CLC legend with {len(label2_mapping)} land cover classes")
    except Exception as e:
        print(f"Warning: Could not load CLC legend from {clc_legend_path}: {e}")
        label2_mapping = {}
        label3_mapping = {}
    
    # Load lookup table to count wells per land cover class
    try:
        lookup_df = pd.read_csv(lookup_table_path)
        
        # Determine which land cover column exists
        if 'dominant_land_cover_2018' in lookup_df.columns:
            landcover_col = 'dominant_land_cover_2018'
        elif 'CLC_2018Code_18' in lookup_df.columns:
            landcover_col = 'CLC_2018Code_18'
        else:
            raise KeyError(f"Neither 'dominant_land_cover_2018' nor 'CLC_2018Code_18' found in lookup table. Available columns: {list(lookup_df.columns)}")
        
        # Ensure land cover values are strings for consistent comparison
        lookup_df[landcover_col] = lookup_df[landcover_col].astype(str)
        
        # Count wells per land cover class
        well_counts = lookup_df[landcover_col].value_counts().to_dict()
        
        # Also create a normalized version (removing .0 suffixes) for matching
        well_counts_normalized = {}
        for key, value in well_counts.items():
            key_clean = str(key).replace('.0', '').strip()
            well_counts_normalized[key_clean] = value
        
        print(f"Loaded lookup table with {len(lookup_df)} wells")
        print(f"Found {len(well_counts)} unique land cover classes in metadata")
    except Exception as e:
        print(f"Warning: Could not load lookup table from {lookup_table_path}: {e}")
        well_counts = {}
        well_counts_normalized = {}
    
    print(f"Loading SHAP values by landcover from: {txt_path}")
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # Parse the file to extract landcover sections
    landcover_data = {}
    current_landcover = None
    current_section = None
    current_n_samples = None
    shap_lines = []
    input_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip comments and empty lines at the start
        if line.startswith('#') or line == '':
            i += 1
            continue
        
        # Check for landcover identifier
        if line.startswith('landcover_'):
            # Save previous landcover if exists
            if current_landcover is not None and shap_lines and input_lines:
                shap_vals = np.array([list(map(float, l.strip().split())) for l in shap_lines])
                input_data = np.array([list(map(float, l.strip().split())) for l in input_lines])
                # Use n_samples from file if available, otherwise it will be set based on number of wells
                landcover_data[current_landcover] = {
                    'shap_vals': shap_vals,
                    'input_data': input_data,
                    'n_samples': current_n_samples  # Will be set later if None
                }
            
            # Start new landcover
            current_landcover = line.replace('landcover_', '')
            current_section = None
            current_n_samples = None
            shap_lines = []
            input_lines = []
        
        # Check for section markers
        elif line == 'shap_vals':
            current_section = 'shap'
        elif line == 'input_data':
            current_section = 'input'
        elif line.startswith('n_samples_'):
            # Extract sample count
            try:
                current_n_samples = int(line.replace('n_samples_', '').strip())
            except:
                current_n_samples = None
        elif current_section == 'shap' and line:
            shap_lines.append(line)
        elif current_section == 'input' and line:
            input_lines.append(line)
        
        i += 1
    
    # Don't forget the last landcover
    if current_landcover is not None and shap_lines and input_lines:
        shap_vals = np.array([list(map(float, l.strip().split())) for l in shap_lines])
        input_data = np.array([list(map(float, l.strip().split())) for l in input_lines])
        landcover_data[current_landcover] = {
            'shap_vals': shap_vals,
            'input_data': input_data,
            'n_samples': current_n_samples  # Will be set later if None
        }
    
    print(f"Found {len(landcover_data)} landcover classes to plot")
    
    # Plot each landcover class
    feature_names = custom_labels if custom_labels is not None else columns_to_keep
    
    for landcover, data in landcover_data.items():
        # Get n_samples (number of wells) from the lookup table based on land cover code
        landcover_code = str(landcover).strip()
        landcover_code_clean = landcover_code.replace('.0', '').strip()
        
        # Try to find the well count for this land cover class
        n_samples = None
        if landcover_code in well_counts:
            n_samples = well_counts[landcover_code]
        elif landcover_code_clean in well_counts_normalized:
            n_samples = well_counts_normalized[landcover_code_clean]
        elif landcover_code_clean in well_counts:
            n_samples = well_counts[landcover_code_clean]
        
        if n_samples is None:
            print(f"Warning: Could not find well count for landcover {landcover_code} in metadata. Setting n_samples to 0.")
            n_samples = 0
        
        print(f"Plotting SHAP values for landcover: {landcover_code} (n={n_samples} wells)")
        
        # Decode land cover name from CLC code (use already cleaned code)
        landcover_name = landcover_code_clean  # Default to code if not found
        
        # Try to find in legend mappings
        if landcover_code_clean in label2_mapping:
            landcover_name = label2_mapping[landcover_code_clean]
        elif landcover_code_clean in label3_mapping:
            landcover_name = label3_mapping[landcover_code_clean]
        elif landcover_code in label2_mapping:
            landcover_name = label2_mapping[landcover_code]
        elif landcover_code in label3_mapping:
            landcover_name = label3_mapping[landcover_code]
        
        try:
            # Create new figure for each landcover
            plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
            
            shap.summary_plot(data['shap_vals'], data['input_data'], 
                            feature_names=feature_names, show=False)
            
            plt.xlabel("SHAP value", fontsize=15)
            # Include land cover name and sample count in title
            plt.title(f"SHAP Values - {landcover_name}\n(CLC Code: {landcover_code}, n={n_samples})", 
                     fontsize=16)
            
            # Make horizontal grid lines dotted and linewidth 1
            ax = plt.gca()
            ax.yaxis.grid(True, linestyle=':', linewidth=1)
            
            # Save plot
            # Sanitize landcover name for filename
            safe_landcover = str(landcover).replace('/', '_').replace(' ', '_').replace('\\', '_')
            plot_filename = f"{model_dir}/shap_values_landcover_{safe_landcover}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"  Saved plot to: {plot_filename}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"[WARNING] Plotting error for landcover '{landcover}': {e}")
            plt.close()
            continue
    
    print(f"Completed plotting SHAP values for {len(landcover_data)} landcover classes")


def plot_r2_rmse_boxplots(df, model_dir, fontsize=16, show_plot=False):
    """
    Plots boxplots for R2 and RMSE from a scores DataFrame and saves the figure.
    Args:
        df (pd.DataFrame): DataFrame with 'R2' and 'RMSE' columns.
        model_dir (str): Directory to save the plot.
        fontsize (int): Font size for plot labels.
        show_plot (bool): Whether to display the plot with plt.show().
    """
    import seaborn as sns
    fig, ax = plt.subplots(1, 2, figsize=(8, 5))

    # R2 boxplot
    sns.boxplot(
        y=df['R2'], width=0.5, color="#FFD7DF",
        ax=ax[0],
        boxprops=dict(edgecolor="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
        medianprops=dict(color="#244062", linewidth=2)
    )
    # Show R2 summary statistics on the plot
    s = f"R2 \nmedian = {round(df['R2'].median(),2)}\nmax = {round(df['R2'].max(),2)}\nmin = {round(df['R2'].min(),2)}"
    ax[0].text(0.6, -0.025, s, bbox=dict(facecolor='white'), fontsize=fontsize)
    ax[0].set_ylabel("R2", fontsize=fontsize)
    ax[0].tick_params(axis='both', labelsize=fontsize-2)

    # RMSE boxplot
    sns.boxplot(
        y=df['RMSE'], width=0.5, color="#FFD7DF",
        ax=ax[1],
        boxprops=dict(edgecolor="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
        medianprops=dict(color="#244062", linewidth=2)
    )
    # Show RMSE summary statistics on the plot
    s = f"RMSE \nmedian = {round(df['RMSE'].median(),2)}\nmax = {round(df['RMSE'].max(),2)}\nmin = {round(df['RMSE'].min(),2)}"
    ax[1].text(0.6, 0.035, s, bbox=dict(facecolor='white'), fontsize=fontsize)
    ax[1].set_ylabel("RMSE", fontsize=fontsize)
    ax[1].tick_params(axis='both', labelsize=fontsize-2)

    plt.tight_layout()
    plt.savefig(f"{model_dir}/boxplot_r2_rmse.png", dpi=300)
    if show_plot:
        plt.show()
    plt.close()


def plot_r2_rmse_nse_bias_boxplot(df, model_dir, fontsize=16, show_plot=False):
    """
    Plots boxplots for R2, RMSE, NSE, and Bias from a scores DataFrame and saves the figure.
    Args:
        df (pd.DataFrame): DataFrame with 'R2', 'RMSE', 'NSE', and 'Bias' columns.
        model_dir (str): Directory to save the plot.
        fontsize (int): Font size for plot labels.
        show_plot (bool): Whether to display the plot with plt.show().
    """
    import seaborn as sns
    
    # Check which metrics are available
    available_metrics = []
    metric_info = {
        'R2': {'color': '#FFD7DF', 'ylabel': 'R²'},
        'RMSE': {'color': '#B8E6B8', 'ylabel': 'RMSE'},
        'NSE': {'color': '#FFE5B4', 'ylabel': 'NSE'},
        'Bias': {'color': '#D4B5FF', 'ylabel': 'Bias'}
    }
    
    for metric in ['R2', 'RMSE', 'NSE', 'Bias']:
        if metric in df.columns:
            available_metrics.append(metric)
    
    if not available_metrics:
        print("Warning: No valid metrics found in DataFrame")
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 5))
    
    # Handle single metric case
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        # Create boxplot
        sns.boxplot(
            y=df[metric], width=0.5, color=metric_info[metric]['color'],
            ax=axes[i],
            boxprops=dict(edgecolor="black", linewidth=1.5),
            whiskerprops=dict(color="black", linewidth=1.5),
            capprops=dict(color="black", linewidth=1.5),
            medianprops=dict(color="#244062", linewidth=2)
        )
        
        # Calculate statistics
        median_val = df[metric].median()
        max_val = df[metric].max()
        min_val = df[metric].min()
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        
        # Position text box based on data range
        y_range = max_val - min_val
        if metric == 'RMSE':
            text_y = min_val + 0.1 * y_range
        elif metric == 'Bias':
            text_y = max_val - 0.3 * y_range
        else:  # R2, NSE
            text_y = min_val + 0.1 * y_range
        
        # Create summary statistics text
        stats_text = (f"{metric}\n"
                     f"median = {median_val:.3f}\n"
                     f"mean = {mean_val:.3f}\n"
                     f"std = {std_val:.3f}\n"
                     f"max = {max_val:.3f}\n"
                     f"min = {min_val:.3f}")
        
        axes[i].text(0.6, text_y, stats_text, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'), 
                    fontsize=fontsize-4, verticalalignment='bottom')
        
        axes[i].set_ylabel(metric_info[metric]['ylabel'], fontsize=fontsize)
        axes[i].tick_params(axis='both', labelsize=fontsize-2)
        
        # Add reference lines for specific metrics
        if metric == 'R2':
            axes[i].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1)
            axes[i].axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, linewidth=1)
        elif metric == 'NSE':
            axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
            axes[i].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, linewidth=1)
        elif metric == 'Bias':
            axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f"{model_dir}/boxplot_r2_rmse_nse_bias.png", dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    
    print(f"Saved comprehensive boxplot to: {model_dir}/boxplot_r2_rmse_nse_bias.png")


def plot_r2_rmse_boxplots_by_landcover(scores_df, lookup_table_path, model_dir, fontsize=14, show_plot=False, 
                                      clc_legend_path="/Users/marie-christineckert/Nextcloud/TU/phd_work/GW_preprocessing_BB_all/meta_data/CORINE_LandCover/clc_legend.csv"):
    """
    Plots boxplots for R2, RMSE, NSE, and Bias stratified by land cover class using LABEL2 (first 2 digits of CLC code).
    Creates a 4-row layout where each row shows one metric across all land cover classes.
    
    Args:
        scores_df (pd.DataFrame): DataFrame with 'ID', 'R2', 'RMSE', 'NSE', 'Bias' columns.
        lookup_table_path (str): Path to the lookup table CSV with ID and dominant_land_cover_2018 columns.
        model_dir (str): Directory to save the plot.
        fontsize (int): Font size for plot labels.
        show_plot (bool): Whether to display the plot with plt.show().
        clc_legend_path (str): Path to the CLC legend CSV file with CLC_CODE and LABEL2 columns.
    """
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Load lookup table and CLC legend
    lookup_df = pd.read_csv(lookup_table_path)
    clc_legend_df = pd.read_csv(clc_legend_path)
    
    # Ensure ID columns are the same type (convert both to string for consistent merging)
    scores_df['ID'] = scores_df['ID'].astype(str)
    lookup_df['ID'] = lookup_df['ID'].astype(str)
    
    # Determine which land cover column exists in the lookup table
    if 'dominant_land_cover_2018' in lookup_df.columns:
        landcover_col = 'dominant_land_cover_2018'
    elif 'CLC_2018Code_18' in lookup_df.columns:
        landcover_col = 'CLC_2018Code_18'
    else:
        raise KeyError(f"Neither 'dominant_land_cover_2018' nor 'CLC_2018Code_18' found in lookup table. Available columns: {list(lookup_df.columns)}")
    
    # Merge scores with land cover data
    merged_df = scores_df.merge(lookup_df[['ID', landcover_col]], on='ID', how='inner')
    
    # Clean and prepare land cover data
    merged_df[landcover_col] = merged_df[landcover_col].astype(str)
    merged_df = merged_df[merged_df[landcover_col] != 'nan']  # Remove NaN values
    merged_df = merged_df[merged_df[landcover_col] != '']     # Remove empty strings
    
    # Extract first 2 digits of CLC code to create LABEL2 classification
    def extract_label2_code(clc_code):
        """Extract first 2 digits from CLC code"""
        try:
            # Remove .0 if present and convert to string
            code_str = str(int(float(clc_code)))
            # Take first 2 digits
            if len(code_str) >= 2:
                return code_str[:2]
            else:
                return code_str
        except (ValueError, TypeError):
            return None
    
    merged_df['CLC_LABEL2_CODE'] = merged_df[landcover_col].apply(extract_label2_code)
    
    # Remove rows where LABEL2 code extraction failed
    merged_df = merged_df[merged_df['CLC_LABEL2_CODE'].notna()]
    
    # Create mapping from LABEL2 codes to LABEL2 names using the legend
    clc_legend_df['CLC_CODE_STR'] = clc_legend_df['CLC_CODE'].astype(str)
    clc_legend_df['LABEL2_CODE'] = clc_legend_df['CLC_CODE_STR'].str[:2]
    
    # Get unique LABEL2 mappings
    label2_mapping = clc_legend_df.groupby('LABEL2_CODE')['LABEL2'].first().to_dict()
    
    # Get unique land cover classes (LABEL2 codes) and sort them
    unique_classes = sorted(merged_df['CLC_LABEL2_CODE'].unique())
    n_classes = len(unique_classes)
    
    # Create land cover class labels using the legend
    clc_labels = {}
    for code in unique_classes:
        if code in label2_mapping:
            # Add line breaks for better display
            label = label2_mapping[code]
            if len(label) > 15:  # Break long labels
                words = label.split()
                if len(words) > 2:
                    mid = len(words) // 2
                    label = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
            clc_labels[code] = label
        else:
            clc_labels[code] = f'CLC {code}'
    
    # Define metrics to plot
    metrics = ['R2', 'RMSE', 'NSE', 'Bias']
    metric_colors = {
        'R2': "#FFD7DF",
        'RMSE': "#D7E4FF", 
        'NSE': "#D7FFD7",
        'Bias': "#FFFFD7"
    }
    
    # Create figure with 4 rows (one for each metric) and n_classes columns
    fig, axes = plt.subplots(4, n_classes, figsize=(3*n_classes, 12))
    
    # Ensure axes is always a 2D array for consistent indexing
    if n_classes == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each metric in its own row
    for metric_idx, metric in enumerate(metrics):
        # Check if metric exists in the data
        if metric not in merged_df.columns:
            print(f"Warning: {metric} column not found in data. Skipping.")
            # Hide the entire row
            for col in range(n_classes):
                axes[metric_idx, col].set_visible(False)
            continue
            
        for class_idx, clc_class in enumerate(unique_classes):
            ax = axes[metric_idx, class_idx]
            
            # Filter data for this land cover class using LABEL2 code
            class_data = merged_df[merged_df['CLC_LABEL2_CODE'] == clc_class]
            n_samples = len(class_data)
            
            # Skip if no data
            if n_samples == 0:
                ax.set_visible(False)
                continue
            
            # Get metric data
            metric_data = class_data[metric].values
            
            # Create boxplot
            bp = ax.boxplot([metric_data], widths=0.6, patch_artist=True,
                           boxprops=dict(facecolor=metric_colors[metric], edgecolor="black", linewidth=1.5),
                           whiskerprops=dict(color="black", linewidth=1.5),
                           capprops=dict(color="black", linewidth=1.5),
                           medianprops=dict(color="#244062", linewidth=2))
            
            # Set labels and formatting
            ax.set_xticks([])  # Remove x-axis ticks since we only have one boxplot per subplot
            
            # Add title only for the top row
            if metric_idx == 0:
                class_label = clc_labels.get(clc_class, f'CLC {clc_class}')
                ax.set_title(f'{class_label}\n(n={n_samples})', fontsize=fontsize-2, pad=10)
            
            # Add y-label only for the first column
            if class_idx == 0:
                ax.set_ylabel(metric, fontsize=fontsize, fontweight='bold')
            
            # Add summary statistics as text
            metric_median = np.median(metric_data)
            metric_mean = np.mean(metric_data)
            
            # Position text box
            y_min, y_max = ax.get_ylim()
            text_y = y_min + 0.1 * (y_max - y_min)
            
            ax.text(1, text_y, f'Med: {metric_median:.3f}\nMean: {metric_mean:.3f}', 
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   fontsize=fontsize-4)
            
            ax.tick_params(axis='y', labelsize=fontsize-4)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set consistent y-axis limits for each metric across all land cover classes
            if metric_idx == 0:  # Store y-limits for the first class to use for all
                if class_idx == 0:
                    all_data = []
                    for temp_class in unique_classes:
                        temp_data = merged_df[merged_df['CLC_LABEL2_CODE'] == temp_class]
                        if len(temp_data) > 0 and metric in temp_data.columns:
                            all_data.extend(temp_data[metric].values)
                    if all_data:
                        global_min = np.min(all_data)
                        global_max = np.max(all_data)
                        margin = (global_max - global_min) * 0.1
                        ax.set_ylim(global_min - margin, global_max + margin)
                        # Store limits for other subplots in this row
                        setattr(fig, f'{metric}_ylim', (global_min - margin, global_max + margin))
                else:
                    if hasattr(fig, f'{metric}_ylim'):
                        ax.set_ylim(getattr(fig, f'{metric}_ylim'))
            else:
                # For subsequent metrics, calculate and set consistent limits
                if class_idx == 0:
                    all_data = []
                    for temp_class in unique_classes:
                        temp_data = merged_df[merged_df['CLC_LABEL2_CODE'] == temp_class]
                        if len(temp_data) > 0 and metric in temp_data.columns:
                            all_data.extend(temp_data[metric].values)
                    if all_data:
                        global_min = np.min(all_data)
                        global_max = np.max(all_data)
                        margin = (global_max - global_min) * 0.1
                        ax.set_ylim(global_min - margin, global_max + margin)
                        setattr(fig, f'{metric}_ylim', (global_min - margin, global_max + margin))
                else:
                    if hasattr(fig, f'{metric}_ylim'):
                        ax.set_ylim(getattr(fig, f'{metric}_ylim'))
    
    # Add overall title
    fig.suptitle('Model Performance Metrics by Land Cover Class (CLC 2018 - LABEL2)', fontsize=fontsize+2, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)
    
    # Save figure
    plt.savefig(f"{model_dir}/boxplot_metrics_by_landcover_label2.png", dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    
    # Print summary statistics
    print("Summary by Land Cover Class (LABEL2):")
    print("="*70)
    for clc_class in unique_classes:
        class_data = merged_df[merged_df['CLC_LABEL2_CODE'] == clc_class]
        class_label = clc_labels.get(clc_class, f'CLC {clc_class}').replace('\n', ' ')
        print(f"{class_label} (n={len(class_data)}):")
        for metric in metrics:
            if metric in class_data.columns:
                print(f"  {metric:4s} - Median: {class_data[metric].median():.3f}, Mean: {class_data[metric].mean():.3f}, Std: {class_data[metric].std():.3f}")
        print()
        
    # Print mapping of LABEL2 codes to original CLC codes for reference
    print("\nLABEL2 to Original CLC Code Mapping:")
    print("="*50)
    for clc_class in unique_classes:
        class_data = merged_df[merged_df['CLC_LABEL2_CODE'] == clc_class]
        original_codes = sorted(class_data['dominant_land_cover_2018'].unique())
        class_label = clc_labels.get(clc_class, f'CLC {clc_class}').replace('\n', ' ')
        print(f"{class_label} ({clc_class}): {', '.join(original_codes)}")
    print()

