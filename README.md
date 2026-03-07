# 📈 Customer Lifetime Value (CLV) Prediction with Survival Analysis

> Predict **when** a customer will churn and **how much revenue** they'll generate — going beyond binary churn classification using survival analysis and machine learning.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-red?style=flat-square)
![Plotly Dash](https://img.shields.io/badge/Plotly-Dashboard-636efa?style=flat-square&logo=plotly)
![lifelines](https://img.shields.io/badge/lifelines-Survival%20Analysis-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🎯 What This Project Does

Most churn models answer *"will this customer leave?"* — this project answers *"when will they leave, and what are they worth until then?"*

It combines **Kaplan-Meier survival curves**, **Cox Proportional Hazards regression**, and an **XGBoost CLV regressor** to give businesses a complete picture of customer retention risk and future revenue potential — all accessible through an interactive Plotly Dash dashboard.

---

## 🖥️ Live Dashboard

### Dashboard Overview — Key Metrics
> Avg Lifetime Value: **$121.47** · Churn Rate: **62.0%** · Median Lifetime: **302 Days**

![Dashboard Overview](https://raw.githubusercontent.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis/main/Output_Results/Screenshot%28121%29.png)

The dashboard has 3 tabs — Retention & Churn, Lifetime Value Prediction, and Predict New Customer.

---

## 📊 Output Visualisations

### 1 · Retention Curve by Plan Type

![Retention by Plan Type](https://raw.githubusercontent.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis/main/Output_Results/newplot.png)

---

### 2 · Retention Curve by Region

![Retention by Region](https://raw.githubusercontent.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis/main/Output_Results/newplot%281%29.png)

---

### 3 · Retention by Region — Mid-Term Zoom (Days 330–630)

![Retention by Region Zoomed](https://raw.githubusercontent.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis/main/Output_Results/newplot%282%29.png)

---

### 4 · Retention Curve by Acquisition Channel

![Retention by Channel](https://raw.githubusercontent.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis/main/Output_Results/newplot%283%29.png)

---

### 5 · Feature Importance + Actual vs Predicted CLV

![Feature Importance and Model Performance](https://raw.githubusercontent.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis/main/Output_Results/Screenshot%28122%29.png)

---

### 6 · CLV Simulator — Alternative Input Example

![CLV Simulator EU Basic](https://raw.githubusercontent.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis/main/Output_Results/Screenshot%28124%29.png)

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis.git
cd Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full CLV pipeline (survival analysis + model training)
python main.py

# 4. Launch the interactive dashboard
python dashboard/app.py
```

Open `http://localhost:8051` in your browser.

---

## 🏗️ Project Structure

```
├── main.py                       # Pipeline entry point
├── requirements.txt
├── data/
│   ├── generate_clv_data.py      # Synthetic data generator (20,000+ customers)
│   ├── customers.csv
│   └── transactions.csv
├── src/
│   ├── survival_analysis.py      # Kaplan-Meier & Cox PH models
│   ├── feature_engineering.py    # RFM feature computation
│   └── regression_model.py       # XGBoost CLV regressor + visualizer
├── dashboard/
│   └── app.py                    # Plotly Dash interactive dashboard
├── Output_Results/               # All output screenshots
└── reports/
    ├── analysis_summary.json
    └── technical_report.md
```

---

## 🔬 Methodology

### 1 · Data
Synthetic dataset of **20,000+ customers** across 5 regions (NA, EU, APAC, LATAM, MENA) and 6 acquisition channels, simulating a subscription SaaS business with realistic churn decay patterns.

### 2 · Survival Analysis
- **Kaplan-Meier**: Non-parametric retention curves segmented by plan type, region, and channel
- **Log-Rank Test**: Statistical significance of differences between survival curves
- **Cox Proportional Hazards**: Identifies which covariates drive churn hazard and by how much
- **C-index**: Primary model evaluation metric

### 3 · Feature Engineering (RFM)
| Feature | Description |
|---|---|
| **Recency** | Days since last transaction |
| **Frequency** | Total number of purchases |
| **Monetary** | Total spend to date |
| **Tenure** | Days since first purchase |
| **Avg Order Value** | Mean transaction amount |
| **Purchases/Month** | Purchase rate normalised by tenure |

### 4 · CLV Prediction
- **Model**: XGBoost Regressor
- **Target**: Revenue in the next 365 days (train/test split at Dec 31, 2024)
- **Top predictor**: `plan_type_Enterprise` (importance ~0.8)
- **Metrics**: MAE · R²

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Avg Lifetime Value | $121.47 |
| Overall Churn Rate | 62.0% |
| Median Customer Lifetime | 302 days |
| Best Retention — Plan | Enterprise |
| Best Retention — Channel | Referral |
| Worst Retention — Channel | Ads |
| Top CLV Predictor | plan_type_Enterprise |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Survival Analysis | `lifelines` |
| ML Model | `XGBoost` · `scikit-learn` |
| Data | `pandas` · `numpy` |
| Visualisation | `plotly` |
| Dashboard | `Dash` · `dash-bootstrap-components` |

---

## 💡 Business Insights

- **Enterprise plan** customers have dramatically higher retention and CLV — upselling is the single highest-leverage action
- **Referral** is the best-performing acquisition channel for long-term retention; **Ads** produces customers who churn the fastest
- **Region has minimal impact** on retention — operational focus should be on plan tier and channel mix, not geography
- The CLV simulator allows sales/CS teams to estimate a prospect's value before onboarding

---







