# Explainable Prediction of Public-Sector Employee Turnover Intention

This repository contains the source code, data preprocessing notebooks, and the interactive Streamlit dashboard for the thesis "Explainable Prediction of Public-Sector Employee Turnover Intention".

## Project Structure

- dashboard/: Contains the `app.py` script for the Streamlit dashboard, providing Policy Simulation and Explainable AI (XAI) insights.
- notebooks/: Jupyter Notebooks used for Exploratory Data Analysis (EDA), feature engineering, and model training.
- models/: Serialized Machine Learning models (e.g., XGBoost, LightGBM) used for predictions.
- data/: Contains sample or anonymized datasets required to run the dashboard. Download dataset is 2022_OPM_FEVS_PRDF.csv via https://drive.google.com/file/d/1ekqjSOp_33AmqoVZhd1NZ2FX-h-2LOXS/view?usp=drive_link *(Note: Raw sensitive data is strictly excluded).*

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [pip](https://pip.pypa.io/en/stable/) package manager

### Installation

1. Clone the repository to your local machine:
   bash
   git clone <your-github-repository-url>
   cd <repository-folder-name>

2. (Optional but recommended) Create and activate a virtual environment:
   bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate

3. Install the required dependencies:
   bash
   pip install -r requirements.txt

### Running the Dashboard

To launch the interactive dashboard, run the following command from the root directory of the project:

bash
streamlit run dashboard/app.py
The dashboard will open automatically in your default web browser.

## 🔒 Note on Data Privacy
Due to privacy regulations and academic ethics, the original raw dataset containing Personally Identifiable Information (PII) of public-sector employees is **not included** in this repository. Any data provided herein is either fully anonymized or synthetic, intended solely for demonstrating the functionality of the predictive models and the HR policy simulation dashboard.
