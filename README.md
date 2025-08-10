# Glucose Prediction

This code is a refactor of the code used in [Engineering digital biomarkers of interstitial glucose from noninvasive smartwatches](https://doi.org/10.1038/s41746-021-00465-w). Note there may be minor discrepancies between this code and the original paper due to the addition of vector operations, parallel processing, and updating all of the code to work with the new versions of python libraries. 

This repo contains a machine learning pipeline for predicting glucose levels using wearable sensor data, food logs, and demographic information. This project implements both personalized and population-based models using XGBoost with Random Forest feature selection.

## Overview

This codebase provides a complete pipeline for:
- **Feature Engineering**: Processing wearable sensor data (EDA, temperature, heart rate, accelerometer) and food logs and engineers 69 features
- **Model Training**: Training personalized and population-based glucose prediction models
- **Cross-Validation**: Leave-one-participant-out (LOPOCV) and personalized 50/50 split validation strategies

See the [methods section](https://www.nature.com/articles/s41746-021-00465-w#Sec8) of the original paper for additional information on rationale for design choices. 

## Project Structure

```
glucose-prediction/
├── configs/                    # Configuration files
│   ├── fe_config.yaml         # Feature engineering pipeline config
│   ├── model_loocv.yaml       # Population model config
│   └── model_personalized.yaml # Personalized model config
├── data/                       # Data directory (not in repo)
├── src/
│   ├── glucose_fe/            # Feature engineering pipeline
│   │   ├── cli.py            # Command-line interface
│   │   ├── pipeline.py       # Main pipeline orchestration
│   │   ├── features.py       # Feature computation (Pandas)
│   │   ├── features_polars.py # Feature computation (Polars)
│   │   ├── glucose.py        # Glucose data processing
│   │   ├── hrv.py            # Heart rate variability features
│   │   ├── stress.py         # Stress detection features
│   │   ├── wake.py           # Wake/sleep pattern features
│   │   ├── food.py           # Food log processing
│   │   └── io.py             # Data I/O utilities
│   └── models/                # Model training scripts
│       ├── train_personalized_xgb.py  # Personalized XGBoost training
│       ├── train_population_xgb.py    # Population XGBoost training
│       ├── config.py          # Model configuration
│       └── utils.py           # Model utilities
├── notebooks/                  # Jupyter notebooks
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/brinnaebent/glucose-prediction.git
   cd glucose-prediction
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv gp-venv
   source gp-venv/bin/activate  # On Windows: gp-venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```


## Data Setup

Before running the pipeline, you need to organize your data in the following structure. If downloaded from PhysioNet, it will already be organized in this way:

```
data/
├── 001/                       # Participant ID
│   ├── EDA_001.csv               # Empatica EDA sensor data
│   ├── TEMP_001.csv              # Empatica temperature data
│   ├── HR_001.csv                # Empatica heart rate data
│   ├── ACC_001.csv               # Empatica accelerometer data
│   ├── IBI_001.csv               # Empatica Inter-beat interval data
|   ├── Food_Log_001.csv          # Food logs
│   └── Dexcom_001.csv            # Glucose data
├── 002/                       # Another participant
│   └── ...
└── Demographics.csv           # Participant demographics
```

## Configuration

### Feature Engineering Config (`configs/fe_config.yaml`)

Update the paths in `fe_config.yaml` to match your data directory structure:

```yaml
paths:
  root: "/path/to/your/data"
  medx_dir: "."
  food_logs_dir: "."
  out_dir: "/path/to/output/directory"
  demographics_csv: "Demographics.csv"
```

### Model Configs

Update the data paths in the model configuration files:
- `configs/model_loocv.yaml` for population models
- `configs/model_personalized.yaml` for personalized models

## Usage

### 1. Feature Engineering Pipeline

Run the feature engineering pipeline to process raw sensor data:

```bash
python -m src.glucose_fe.cli --config configs/fe_config.yaml --max-workers 1
```

**Options**:
- `--config`: Path to configuration file (required)
- `--compile-only`: Only compile features without processing (optional)
- `--max-workers`: Number of parallel workers (optional, defaults to all cores)

**Output**: The pipeline generates a compiled dataset at `out/ALL_features_cleaned.parquet`

### 2. Model Training

#### Population Model (Leave-One-Participant-Out Cross-Validation)

```bash
python -m src.models.train_population_xgb --config configs/model_loocv.yaml
```

#### Personalized Model (50/50 Split per Participant)

```bash
python -m src.models.train_personalized_xgb --config configs/model_personalized.yaml
```

## Key Components

### Feature Engineering

- **Sensor Features**: Rolling statistics (mean, std, min, max) for EDA, temperature, heart rate, and accelerometer data
- **HRV Features**: Heart rate variability metrics computed over sliding windows
- **Stress Detection**: EDA peak counting for stress level assessment
- **Wake/Sleep Patterns**: Activity-based sleep pattern detection
- **Food Features**: Meal timing and nutritional information processing

### Modeling

- **Feature Selection**: Random Forest-based feature importance filtering
- **XGBoost Models**: Gradient boosting with early stopping
- **Validation Strategies**: 
  - Population: Leave-one-participant-out cross-validation
  - Personalized: 50/50 temporal split per participant

## Performance Optimization

- **Polars Engine**: Use `engine: "polars"` in config for faster data processing
- **Parallel Processing**: Adjust `max-workers` based on your system capabilities
- **Memory Management**: Polars provides better memory efficiency for large datasets

## Output Structure

The pipeline generates the following outputs:

```
out/
├── ALL_features_cleaned.parquet    # Compiled feature dataset
├── modeling_population/            # Population model outputs
│   ├── models/                     # Trained models
│   ├── preds/                      # Predictions
│   ├── feature_lists/              # Selected features per fold
│   └── feature_importances/        # Feature importance rankings
└── modeling_personalized/          # Personalized model outputs
    ├── models/                     # Trained models
    ├── preds/                      # Predictions
    ├── feature_lists/              # Selected features per participant
    └── feature_importances/        # Feature importance rankings
```

## Results
Using data directly downloaded from [PhysioNet](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/)
And the defaults and configs currently in this repository (as of 8/10/25):

##### Results for Population LOOCV model
Mean RMSE: 22.973 ± 4.767
Mean MAPE: 15.58% ± 4.17%
Mean Accuracy: 84.42% ± 4.17%

##### Results for Personalized model
Mean RMSE: 22.286 ± 4.448
Mean MAPE: 14.07% ± 2.75%
Mean ACC : 85.93% ± 2.75%

Note: small discrepancies from paper exist due to minor changes mentioned above.

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `max-workers` or use Polars engine
2. **Missing Data**: Ensure all required sensor files are present for each participant
3. **Path Errors**: Verify all paths in configuration files are correct and accessible


## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[MIT](https://opensource.org/license/mit)


## Citation

If you use this code in your research, please cite the original paper:

Bent, B., Cho, P.J., Henriquez, M. et al. Engineering digital biomarkers of interstitial glucose from noninvasive smartwatches. npj Digit. Med. 4, 89 (2021). https://doi.org/10.1038/s41746-021-00465-w

## Issues

Please submit an issue! This was refactored ~5 years after the original paper code was written, so there may be minor discrepancies. Support the open source nature of this code - if you fix something, submit a PR and help everyone out :)