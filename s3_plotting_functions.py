# --- Imports and setup ---
import os
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import shap  # For SHAP values

# Import project modules
from s1_data_preparation import *  # Custom data preparation utilities
from s2_model_utils import *  # Custom model utilities

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
def simulate_plot_wells_testset_conf_int(well_ids, TestData_dict, GLOBAL_SETTINGS, model_path, inimax, number_of_wells, BL_abbr = "BB", save_fig = False):
    """ 
    Analyzes test data for a specified number of wells using a trained CNN model.
     inspired and adapted from Wunsch et al 2022 on github https://github.com/AndreasWunsch/Long-Term-GWL-Simulations

    Parameters:
        well_ids (list): List of well IDs.
        TestData_dict (dict): Dictionary containing test data and observation dataframes.
        GLOBAL_SETTINGS (dict): Global settings for the analysis.
        model_path (str): Path to the trained model weights.
        scaler_y (object): Scaler for inverse transformation of predictions.
        predict_distribution (function): Function to predict distributions using the model.
        number_of_wells (int): Number of wells to analyze.
        inimax (int): Number of simulation runs for uncertainty analysis.

    Returns:
        dict: Dictionary containing scores for each well.
    """
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
            loaded_model = tf.keras.models.load_model(model_path + f'model_weights_{BL_abbr}_run_{ini}.keras')
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

def plot_shap_from_txt(txt_path, columns_to_keep, show_plot=True):
    """
    Loads SHAP values and input data from a .txt file and plots a SHAP summary plot.
    Args:
        txt_path: Path to the .txt file containing SHAP values and input data.
        columns_to_keep: List of feature names.
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
        shap.summary_plot(shap_vals, input_data, feature_names=columns_to_keep, show=False)
        plt.xlabel("SHAP value", fontsize=15)
        plt.savefig(f"{model_dir}/shap_values_plot.png", dpi=300)
        if show_plot:
            plt.show()
        plt.close()
    except Exception as e:
        print(f"[WARNING] Plotting error in plot_shap_from_txt: {e}")



# example usage
# plot_shap_from_txt(
#     txt_path=os.path.join(model_dir, f"shapvalues.txt"),
#     columns_to_keep=columns_to_keep,
#     show_plot=True
# )