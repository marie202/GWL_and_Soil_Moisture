"""
Optuna hyperparameter optimization for the CNN (no LSTM) model used in `02_CNN_SM.py`.
Uses all available well files (`data_217/*.csv`) and searches over window size, dense size,
batch size, and number of filters.
"""

# --- Imports ---
import os
import glob
import json
import pickle
import random
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Project modules
from s1_data_preparation import process_data_pipeline, scaler_statics_global, scale_dataset_indiv
from s2_model_utils import build_cnn_model

# Reproducibility
tf.random.set_seed(1 + 63493)
np.random.seed(1 + 347823)
random.seed(1 + 347823)

# --- Paths and data ---
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = str(BASE_DIR / "data_217/*.csv")
ALL_FILES = glob.glob(INPUT_DIR)

print(f"Using ALL {len(ALL_FILES)} files for optimization (pattern: {INPUT_DIR})")

# Feature configuration (matches 02_CNN_SM.py)
COLUMNS_TO_KEEP = [
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

STATIC_COLS = [
    "elevation_msl",
    "MW_muGOK",
    "distance_to_waterwork_km",
    "kf_remap_number",
]

# Base settings that do not change across trials
GLOBAL_SETTINGS_BASE = {
    "inimax": 3,  # ensemble members per trial (kept low for speed)
    "kernel_size": 3,
    "clip_norm": True,
    "clip_value": 1,
    "epochs": 200,
    "learning_rate": 1e-5,
    "test_start": pd.to_datetime("2019-01-01", format="%Y-%m-%d"),
    "test_end": pd.to_datetime("2024-12-31", format="%Y-%m-%d"),
    "num_cnn_layers": 6,
    "use_cnn_only": True,  # ensure CNN-only architecture
    "model_dir_note": f"optuna_cnn_allfiles_{len(ALL_FILES)}wells_v1",
}


# --- Objective function ---
def objective(trial: optuna.Trial) -> float:
    # Hyperparameters to search
    densesize_int = trial.suggest_int("densesize", 64, 256, step=8)
    windowsize_int = trial.suggest_int("windowsize", 30, 80, step=2)
    batchsize_int = trial.suggest_categorical(
        "batchsize", [64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512]
    )
    filters_int = trial.suggest_int("filters", 32, 192, step=8)

    print(f"\nTrial {trial.number}: densesize={densesize_int}, windowsize={windowsize_int}, "
          f"batchsize={batchsize_int}, filters={filters_int}")

    # Compose settings for this trial
    GLOBAL_SETTINGS = {
        **GLOBAL_SETTINGS_BASE,
        "batch_size": batchsize_int,
        "dense_size": densesize_int,
        "filters": filters_int,
        "window_size": windowsize_int,
        "trial": trial,
    }

    try:
        # Static scaler
        scaler_static, _ = scaler_statics_global(
            input_dir=INPUT_DIR,
            static_cols=STATIC_COLS,
            columns_to_keep=COLUMNS_TO_KEEP,
        )
        print(f"✓ Trial {trial.number}: Static scaler prepared")

        # Data pipeline (scaled + split)
        X_train, Y_train, X_val, Y_val, X_opt, Y_opt, ScalerData_dict, ValData_dict, OptData_dict, TestData_dict = (
            process_data_pipeline(
                input_dir=INPUT_DIR,
                columns_to_keep=COLUMNS_TO_KEEP,
                static_cols=STATIC_COLS,
                GLOBAL_SETTINGS=GLOBAL_SETTINGS,
                scaler_static=scaler_static,
                target_column="GWL",
            )
        )
        print(f"✓ Trial {trial.number}: Data ready — X_train {X_train.shape}, X_opt {X_opt.shape}")

        # Build scaler_y for inverse transform
        scaler_data = ScalerData_dict[list(ScalerData_dict.keys())[0]].iloc[GLOBAL_SETTINGS["window_size"] :]
        _, scaler_y, _ = scale_dataset_indiv(scaler_data, target_column="GWL")

        inimax = GLOBAL_SETTINGS["inimax"]
        opt_sim_members = np.zeros((len(X_opt), inimax))

        # Ensemble training
        for ini in range(inimax):
            print(f"  → Ensemble member {ini + 1}/{inimax}")
            model, history = build_cnn_model(
                ini, GLOBAL_SETTINGS, X_train, Y_train, X_val, Y_val
            )

            # Pruning point after training this member
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            opt_sim_n = model.predict(X_opt)
            opt_sim = scaler_y.inverse_transform(opt_sim_n)
            opt_sim_members[:, ini] = opt_sim.reshape(-1,)

            # Early pruning based on first member performance
            if ini == 0:
                temp_sim = np.asarray(opt_sim.reshape(-1, 1))
                temp_obs = np.asarray(scaler_y.inverse_transform(Y_opt.reshape(-1, 1)))

                temp_r = stats.linregress(temp_sim[:, 0], temp_obs[:, 0])
                temp_r2 = temp_r.rvalue ** 2
                temp_obs_mean = np.mean(temp_obs)
                temp_numerator = np.sum((temp_obs - temp_sim) ** 2)
                temp_denominator = np.sum((temp_obs - temp_obs_mean) ** 2)
                temp_nse = 1 - (temp_numerator / temp_denominator)

                temp_r2_clipped = max(temp_r2, 0.001)
                temp_nse_clipped = max(temp_nse, 0.001)
                temp_score = np.sqrt(temp_r2_clipped * temp_nse_clipped)

                trial.report(temp_score, ini)
                if trial.should_prune():
                    trial.set_user_attr("early_stop_reason", "optuna_pruner")
                    trial.set_user_attr("stopped_at_ensemble_member", ini + 1)
                    raise optuna.exceptions.TrialPruned()

        # Aggregate ensemble predictions (median)
        opt_sim_median = np.median(opt_sim_members, axis=1)
        sim = np.asarray(opt_sim_median.reshape(-1, 1))
        obs = np.asarray(scaler_y.inverse_transform(Y_opt.reshape(-1, 1)))

        r = stats.linregress(sim[:, 0], obs[:, 0])
        r2 = r.rvalue ** 2

        obs_mean = np.mean(obs)
        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - obs_mean) ** 2)
        nse = 1 - (numerator / denominator)

        r2_clipped = max(r2, 0.001)
        nse_clipped = max(nse, 0.001)
        score = np.sqrt(r2_clipped * nse_clipped)

        err = sim - obs
        RMSE = np.sqrt(np.mean(err ** 2))
        obs_range = np.max(obs) - np.min(obs)
        nRMSE = RMSE / obs_range

        trial.set_user_attr("r2", float(r2))
        trial.set_user_attr("nse", float(nse))
        trial.set_user_attr("rmse", float(RMSE))
        trial.set_user_attr("nrmse", float(nRMSE))

        print(f"Trial {trial.number} results: R²={r2:.3f}, NSE={nse:.3f}, Score={score:.3f}, "
              f"RMSE={RMSE:.3f}, nRMSE={nRMSE:.3f}")

        return score

    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial.number} pruned")
        raise
    except Exception as e:
        print(f"[ERROR] Trial {trial.number} failed: {e}")
        return 0.0


# --- Logging helpers ---
def save_iteration_result(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
    if trial.value is not None:
        duration_min = trial.duration.total_seconds() / 60 if trial.duration else 0
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        output = f"[{current_time}] Trial {trial.number}: Score={trial.value:.6f}"

        metrics = []
        if "r2" in trial.user_attrs:
            metrics.append(f"R²={trial.user_attrs['r2']:.3f}")
        if "nse" in trial.user_attrs:
            metrics.append(f"NSE={trial.user_attrs['nse']:.3f}")
        if metrics:
            output += f" ({', '.join(metrics)})"

        output += f", Duration={duration_min:.1f}min, State={trial.state.name}"
        print(output)
    else:
        duration_min = trial.duration.total_seconds() / 60 if trial.duration else 0
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] Trial {trial.number}: Duration={duration_min:.1f}min, {trial.state.name}")


def save_study_progress(study: optuna.study.Study, model_dir: Path) -> None:
    all_trials = []
    for trial in study.trials:
        info = {
            "trial_number": trial.number,
            "value": trial.value if trial.value is not None else None,
            "params": trial.params,
            "state": trial.state.name,
            "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
        }
        if trial.user_attrs:
            info["metrics"] = {k: v for k, v in trial.user_attrs.items() if k in {"r2", "nse", "rmse", "nrmse"}}
        all_trials.append(info)

    completed = [t for t in all_trials if t["value"] is not None]
    completed.sort(key=lambda x: x["value"], reverse=True)
    failed = [t for t in all_trials if t["value"] is None]

    all_trials_file = model_dir / "all_trials_log.json"
    with open(all_trials_file, "w") as f:
        json.dump(
            {
                "last_updated": datetime.datetime.now().isoformat(),
                "summary": {
                    "total_trials": len(study.trials),
                    "completed_trials": len(completed),
                    "pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                    "failed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
                },
                "best_trial": {
                    "number": study.best_trial.number if study.best_trial else None,
                    "value": study.best_value if study.best_trial else None,
                    "params": study.best_params if study.best_trial else None,
                }
                if study.best_trial
                else None,
                "all_trials_ranked": completed + failed,
            },
            f,
            indent=2,
        )
    print(f"📝 Saved study progress to {all_trials_file}")


def create_study() -> optuna.study.Study:
    pruner = MedianPruner(n_startup_trials=25, n_warmup_steps=5, interval_steps=1)
    sampler = TPESampler(seed=42)
    return optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)


# --- Main run ---
if __name__ == "__main__":
    print("[Optuna] Starting CNN hyperparameter optimization...")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = BASE_DIR / "model_runs" / "OptunaOpt" / f"OptunaOpt_{timestamp}_CNN_{GLOBAL_SETTINGS_BASE['num_cnn_layers']}layer_{len(COLUMNS_TO_KEEP)}params_{GLOBAL_SETTINGS_BASE['model_dir_note']}"
    os.makedirs(model_dir, exist_ok=True)
    print(f"Results will be saved to: {model_dir}")

    study = create_study()

    def trial_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        save_iteration_result(study, trial)
        save_study_progress(study, model_dir)

    n_trials = 150
    print(f"[Optuna] Running for up to {n_trials} trials...")

    try:
        study.optimize(objective, n_trials=n_trials, timeout=None, callbacks=[trial_callback], show_progress_bar=True)
        print("\n[Optuna] Optimization completed successfully!")
    except KeyboardInterrupt:
        print("\n[Optuna] Optimization interrupted by user")
    except Exception as e:
        print(f"\n[Optuna] Optimization failed with error: {e}")

    # Always save final progress
    print("\n[Optuna] Saving final study progress...")
    save_study_progress(study, model_dir)

    # Save best params
    if study.best_trial:
        best = {
            "timestamp": datetime.datetime.now().isoformat(),
            "best_trial": {
                "number": study.best_trial.number,
                "score": study.best_value,
                "params": study.best_params,
            },
            "summary": {
                "n_trials": len(study.trials),
                "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                "n_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            },
        }
        best_params_file = model_dir / "best_params.json"
        with open(best_params_file, "w") as f:
            json.dump(best, f, indent=2)
        print(f"✓ Best params saved to: {best_params_file}")
    else:
        print("No successful trials to save best params.")

    # Append to master log
    master_log_file = BASE_DIR / "model_runs" / "OptunaOpt" / "optimization_runs_log.json"
    os.makedirs(master_log_file.parent, exist_ok=True)
    if os.path.exists(master_log_file):
        with open(master_log_file, "r") as f:
            master_log = json.load(f)
    else:
        master_log = {"optimization_runs": []}

    if len(study.trials) > 0 and any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        run_entry = {
            "timestamp": timestamp,
            "directory": str(model_dir),
            "best_score": study.best_value,
            "best_params": dict(study.best_params),
            "n_trials": len(study.trials),
            "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "n_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            "files_used": len(ALL_FILES),
            "optimization_type": "cnn_all_files",
        }
    else:
        run_entry = {
            "timestamp": timestamp,
            "directory": str(model_dir),
            "status": "failed - no successful trials",
            "n_trials": len(study.trials),
            "files_used": len(ALL_FILES),
            "optimization_type": "cnn_all_files",
        }

    master_log["optimization_runs"].append(run_entry)
    with open(master_log_file, "w") as f:
        json.dump(master_log, f, indent=2)

    print(f"📝 Added run to master log: {master_log_file}")
    print(f"📁 This run's results: {model_dir}")
    print(f"🕐 Total optimization runs logged: {len(master_log['optimization_runs'])}")

