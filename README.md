# Explainable Prediction of Public-Sector Employee Turnover Intention

This repository contains the source code, data preprocessing notebooks, and the interactive Streamlit dashboard for the thesis "Explainable Prediction of Public-Sector Employee Turnover Intention".

## 📂 Project Structure

- `dashboard/`: Contains the `app.py` script for the Streamlit dashboard, providing Policy Simulation and Explainable AI (XAI) insights.
- `notebooks/`: Jupyter Notebooks used for Exploratory Data Analysis (EDA), feature engineering, and model training.
- `models/`: Serialized Machine Learning models (e.g., XGBoost, LightGBM) used for predictions.
- `data/`: Contains sample or anonymized datasets required to run the dashboard. *(Note: Large datasets and raw sensitive data are excluded from the repository).*

## 📊 Dataset & Setup

### 1. Download the Dataset
To run the dashboard or execute the notebooks, download the preprocessed dataset:
* **Preprocessed Dataset (2022):** Download [2022_OPM_FEVS_PRDF.csv](https://drive.google.com/file/d/1ekqjSOp_33AmqoVZhd1NZ2FX-h-2LOXS/view?usp=drive_link) (Google Drive) and place it inside the `data/` directory.

### 2. Original / Other Years Dataset
If you want to train models or run predictions for other years:
* **Original FEVS Public Release Data Files:** Access them from the [OPM FEVS Public Data Page](https://www.opm.gov/fevs/public-data-file/).
* **Generalizing to other years:** You can download files for other years, run the corresponding data preprocessing notebooks in the `notebooks/` folder, and train new serialized models.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- [pip](https://pip.pypa.io/en/stable/) package manager

### Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/s4rawut/Explainable-Prediction-of-Public-Sector-Employee-Turnover-Intention.git
   cd Explainable-Prediction-of-Public-Sector-Employee-Turnover-Intention
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Dashboard

To launch the interactive dashboard, run the following command from the root directory of the project:

```bash
streamlit run dashboard/app.py
```
The dashboard will open automatically in your default web browser.

### Reproducing the Pipeline (Model Training & Analysis)

If you want to preprocess the data, train the models, or perform explanation analysis from scratch, execute the notebooks in the `notebooks/` directory in the following sequential order:

1. **`01_Data_Exploration.ipynb`** – Exploratory Data Analysis (EDA) to understand the dataset's features and target variable.
2. **`02_Data_Preprocessing.ipynb`** – Data cleaning, handling missing values, encoding, and feature engineering.
3. **`03_Modeling.ipynb`** – Training, evaluation, and serialization (saving) of the machine learning models.
4. **`04_Model_Interpretation.ipynb`** – Explainable AI (XAI) calculations using SHAP to interpret model predictions.

After running these notebooks, the trained model files will be saved in the `models/` directory, which the Streamlit dashboard (`app.py`) uses for predictions.

## 🔒 Note on Data Privacy

Due to privacy regulations and academic ethics, the original raw dataset containing Personally Identifiable Information (PII) of public-sector employees is **not included** in this repository. Any data provided herein is either fully anonymized or synthetic, intended solely for demonstrating the functionality of the predictive models and the HR policy simulation dashboard.
