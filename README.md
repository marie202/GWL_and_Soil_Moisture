
# Groundwater Level and Soil Moisture Modeling with Deep Learning

This repository implements a deep learning-based approach to simulate and analyze long-term groundwater level (GWL) dynamics, leveraging soil moisture and climate data, and static parameters. 
The model utilizes a Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) hybrid architecture for time series prediction.


doi of [Groundwater Level Data – Brandenburg, Germany (Zenodo)](https://doi.org/10.5281/zenodo.17233232)

---
## Author & Contact

**Marie-Christin Eckert**  
*Groundwater modeling and data science enthusiast*  
Feel free to reach out via [LinkedIn](https://www.linkedin.com/in/marie-christin-eckert-16a4ba29a) or [email](mailto:m.eckert@tu-berlin.de)

ORCIDs of authors:
* M.-C. Eckert: [0009-0005-4003-6416](https://orcid.org/0009-0005-4003-6416)
* A. Rudolph: [0000-0002-7368-5018](https://orcid.org/0000-0002-7368-5018)

---

## Project Structure

```
GWL_and_Soil_Moisture/
│
├── 1_1_CNN_LSTM.py                  # Main CNN-LSTM training script
├── 1_2_OPTUNA_CNN_LSTM.py           # Optuna search for CNN-LSTM
├── 2_1_CNN.py                       # CNN benchmark for soil moisture-only input
├── 2_2_OPTUNA_CNN.py                # Optuna search for CNN-only benchmark
├── 3_1_ANN.py                       # ANN baseline/benchmark experiment
├── 3_2_OPTUNA_ANN.py                # Optuna search for ANN baseline
│
├── s1_data_preparation.py           # Data loading and preprocessing utilities
├── s2_model_utils.py                # Model architecture and training helper functions
├── s3_plotting_functions.py         # Functions for evaluation and visualization
├── s4_optuna_opt.py                 # Optuna optimization utilities
│
├── gw_env.yml                       # Conda environment file with dependencies
├── LICENSE                          # MIT License
└── README.md                        # Project overview
```


## Environment Setup

This project uses **Python 3.12.9**. All dependencies can be installed via the provided conda environment file.

```bash
conda env create -f gw_env.yml
conda activate gw_env
```

---

## Key Libraries Used

* **TensorFlow** & **Keras** - Deep learning framework
* **NumPy**, **Unumpy** - Numerical computing and uncertainty propagation
* **pandas** - Data manipulation and analysis
* **scikit-learn** - Machine learning utilities
* **Matplotlib**, **seaborn** - Data visualization
* **Optuna** - Hyperparameter optimization framework
* **SHAP** - Model interpretability and feature importance
* **SciPy** - Scientific computing 

> See full list in [`gw_env.yml`](gw_env.yml)

---

## How to Run

1. **Train the CNN-LSTM (full feature set)**
   ```bash
   python 1_1_CNN_LSTM.py
   ```

2. **Train the CNN soil-moisture benchmark**
   ```bash
   python 2_1_CNN.py
   ```

3. **Train the ANN baseline**
   ```bash
   python 3_1_ANN.py
   ```

4. **Hyperparameter optimization (Optuna)**
   - CNN-LSTM search:
     ```bash
     python 1_2_OPTUNA_CNN_LSTM.py
     ```
   - CNN-only search:
     ```bash
     python 2_2_OPTUNA_CNN.py
     ```
   - ANN search:
     ```bash
     python 3_2_OPTUNA_ANN.py
     ```

The utility scripts (`s1-s4`) contain modular functions that are imported and used within the main scripts. 

---

## License

This project is open-source under the [MIT License](LICENSE). Credit is due to the original authors of the method and software framework where applicable.

---

## Data Availability


### Groundwater Level Data

- **Processed groundwater level (GWL) time series** from **217 monitoring wells** across Brandenburg, Germany.
- Original data source: [LfU Brandenburg Auskunftsplattform Wasser](https://apw.brandenburg.de).
- Data were interpolated and processed for research applications.
- The original groundwater level data are available free of charge from LfU Brandenburg.
- The processed and interpolated groundwater level time series are published with permission from the local authorities:  
  [https://doi.org/10.5281/zenodo.17233232](https://doi.org/10.5281/zenodo.17233232)

### Model Input Data

All other input datasets required to train the models are available online from the following sources:

- **Soil Moisture Data**  
  DWD (2024a). Daily grids of mean soil moisture under predominant land use for Germany v1.0 [Dataset].  
  [https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/soil_moisture/composite/](https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/soil_moisture/composite/)

- **Climate Data**  
  - *Relative Humidity*:  
    DWD (2024b). Raster data set of mean relative humidity in % for Germany - HYRAS-DE-HURS, version v6.0 [Dataset].  
    [https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/humidity/](https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/humidity/)
  - *Temperature*:  
    DWD (2024c). Raster data set of mean temperature for Germany - HYRAS-DE-TAS, version v6.0 [Dataset].  
    [https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/air_temperature_mean/](https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/air_temperature_mean/)
  - *Precipitation*:  
    DWD (2024d). Raster data set of precipitation sums in mm for Germany - HYRAS-DE-PR, version v6.0 [Dataset].  
    [https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/precipitation/](https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/precipitation/)

- **Static Parameters**
   - *Elevation, Distance to Waterworks, Groundwater Table Depth*
    LfU (2025a). Auskunftsplattform Wasser. Grundwasserstand (gesamt) [Dataset].  
    [https://apw.brandenburg.de](https://apw.brandenburg.de)

   - *Hydraulic Conductivity (kf) Values*  
    Bundesanstalt für Geowissenschaften und Rohstoffe (BGR) & Staatliche Geologische Dienste (SGD): Hydrogeologische Übersichtskarte von Deutschland 1:250,000 (HÜK250). Digitaler Datenbestand, Version 1.0.3 [Dataset], 2019.  
    [https://download.bgr.de/bgr/grundwasser/huek250/shp/huek250.zip](https://download.bgr.de/bgr/grundwasser/huek250/shp/huek250.zip)