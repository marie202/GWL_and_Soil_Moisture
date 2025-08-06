
# Groundwater Level and Soil Moisture Modeling with Deep Learning

This repository implements a deep learning-based approach to simulate and analyze long-term groundwater level (GWL) dynamics, leveraging soil moisture and climate data. The modeling is inspired by [Wunsch et al. (2022)](https://doi.org/10.1038/s41467-022-28770-2) and utilizes a Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) hybrid architecture for time series prediction.

[Andreas Wunsch – Long-Term GWL Simulations](https://github.com/AndreasWunsch/Long-Term-GWL-Simulations)

---

## Project Structure

```

GWL\_and\_Soil\_Moisture/
│
├── 01\_CNN\_LSTM.py               # Main model training script
├── 02\_optihyperparams.py        # Hyperparameter optimization using Bayesian search
│
├── s1\_data\_preparation.py       # Data loading and preprocessing utilities
├── s2\_model\_utils.py            # Model architecture and training helper functions
├── s3\_plotting\_functions.py     # Functions for evaluation and visualization
├── s4\_bayesian\_opt.py           # Bayesian Optimization helper functions
│
├── gw\_hpc\_env\_minimal.yml       # Conda environment file with dependencies
└── README.md                    # Project overview

````


## Environment Setup

This project uses **Python 3.12.9**. All dependencies can be installed via the provided conda environment file.

```bash
conda env create -f gw_hpc_env_minimal.yml
conda activate gw_hpc_env
````

---

## Key Libraries Used

* **TensorFlow** & **Keras** 
* **NumPy**, **Unumpy**
* **pandas** 
* **scikit-learn** 
* **Matplotlib**, **seaborn** 
* **Bayesian Optimization** 
* **SHAP** 
* **SciPy** 

> See full list in [`gw_hpc_env_minimal.yml`](gw_hpc_env_minimal.yml)

---

## How to Run

1. **Train the model**
   Run the main training script:

   ```bash
   python 01_CNN_LSTM.py
   ```

2. **Optimize hyperparameters**
   Run the optimizer:

   ```bash
   python 02_optihyperparams.py
   ```

The utility scripts (`s1-s4`) are imported and used within the main scripts. 




---

## License

This project is open-source under the [MIT License](LICENSE). Credit is due to the original authors of the method and software framework where applicable.

---

## Author

**Marie-Christin Eckert**
*Groundwater modeling and data science enthusiast*
Feel free to reach out or collaborate via [LinkedIn](www.linkedin.com/in/marie-christin-eckert-16a4ba29a) or [email](mailto:m.eckert@tu-berin.de)

