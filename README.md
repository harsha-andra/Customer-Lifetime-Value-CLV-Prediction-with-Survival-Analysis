# 📈 Customer Lifetime Value (CLV) Prediction with Survival Analysis

> Predict **when** a customer will churn and **how much revenue** they'll generate — going far beyond binary churn classification using survival analysis and machine learning.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-red?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-Dashboard-636efa?style=flat-square&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🎯 What This Project Does

Most churn models answer *"will this customer leave?"* — this project answers *"when will they leave, and what are they worth until then?"*

It combines **Kaplan-Meier survival curves**, **Cox Proportional Hazards regression**, and **RFM-based ML models** to give businesses a complete picture of customer retention risk and future revenue potential.

---

## 📊 Output Results & Visualisations

### 1 · Survival Curve — Customer Retention Over Time
> Kaplan-Meier curves showing the probability of a customer remaining active across different segments.

![Survival Curve](https://raw.githubusercontent.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis/main/Output_Results/Screenshot%20(121).png)

---

### 2 · Cox Proportional Hazards — Churn Risk Factors
> Which features most strongly accelerate or delay churn. Hazard ratios > 1 indicate increased risk.

![Cox PH Risk Factors](https://raw.githubusercontent.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis/main/Output_Results/Screenshot%20(122).png)

---

### 3 · RFM Feature Distributions
> Recency, Frequency, and Monetary value spread across the customer base — the foundation of CLV modelling.

![RFM Distributions](https://raw.githubusercontent.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis/main/Output_Results/Screenshot%20(123).png)

---

### 4 · CLV Prediction — Model Performance
> Predicted vs actual CLV values. Evaluated using RMSE, MAE, and R² on a held-out test set.

![Model Performance](https://raw.githubusercontent.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis/main/Output_Results/Screenshot%20(124).png)

---

### 5 · Customer Segmentation by CLV
> Customers clustered into High / Mid / Low value tiers based on predicted lifetime value.

![Customer Segmentation](https://raw.githubusercontent.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis/main/Output_Results/newplot.png)

---

### 6 · Churn Probability Distribution
> Predicted churn probabilities across all 20,000+ customers. High-risk customers are clearly identifiable.

![Churn Probability](https://raw.githubusercontent.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis/main/Output_Results/newplot%20(1).png)

---

### 7 · Feature Importance — XGBoost / Random Forest
> Top predictors of Customer Lifetime Value ranked by model importance score.

![Feature Importance](https://raw.githubusercontent.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis/main/Output_Results/newplot%20(2).png)

---

### 8 · Revenue Forecast by Customer Segment
> Projected 12-month revenue broken down by customer segment to guide retention investment.

![Revenue Forecast](https://raw.githubusercontent.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis/main/Output_Results/newplot%20(3).png)

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/harsha-andra/Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis.git
cd Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full CLV pipeline
python main.py

# 4. Launch the interactive dashboard
python dashboard/app.py
```

Open `http://localhost:8050` in your browser to explore the dashboard.

---

## 🏗️ Project Structure

```
Customer-Lifetime-Value-CLV-Prediction-with-Survival-Analysis/
├── main.py                          # Entry point — runs full pipeline
├── requirements.txt
├── data/
│   └── generate_clv_data.py         # Synthetic data generator (20,000+ customers)
├── src/
│   ├── survival_analysis.py         # Kaplan-Meier & Cox PH models
│   ├── feature_engineering.py       # RFM & behavioural feature engineering
│   ├── regression_model.py          # CLV regression (Random Forest / XGBoost)
│   └── visualization.py             # All charts and survival curve plots
├── dashboard/
│   └── app.py                       # Interactive Plotly Dash retention dashboard
├── Output_Results/                  # All 8 output visualisations
│   ├── Screenshot (121).png         # Kaplan-Meier survival curves
│   ├── Screenshot (122).png         # Cox PH churn risk factors
│   ├── Screenshot (123).png         # RFM feature distributions
│   ├── Screenshot (124).png         # CLV model performance
│   ├── newplot.png                  # Customer segmentation by CLV
│   ├── newplot (1).png              # Churn probability distribution
│   ├── newplot (2).png              # Feature importance
│   └── newplot (3).png              # Revenue forecast by segment
└── reports/
    └── technical_report.md
```

---

## 🔬 Methodology

### Step 1 — Data Generation
Synthetic dataset of **20,000+ customers** across 4 regions (NA, EU, APAC, LATAM) simulating a subscription SaaS business with realistic churn decay patterns and purchase consistency.

### Step 2 — Feature Engineering (RFM)

| Feature | Description |
|---|---|
| **Recency** | Days since last transaction |
| **Frequency** | Number of purchases |
| **Monetary** | Total spend to date |
| **Tenure** | Days since first purchase |
| **Avg Order Value** | Mean transaction size |

### Step 3 — Survival Analysis
- **Kaplan-Meier**: Non-parametric retention curves per customer segment
- **Cox Proportional Hazards**: Identifies which covariates drive churn hazard
- **C-index (Concordance)**: Primary evaluation metric for survival models

### Step 4 — CLV Regression
- Models: **Random Forest** and **XGBoost Regressor**
- Target: Predicted 12-month revenue per customer
- Evaluation: **RMSE · MAE · R²**

---

## 📊 Key Metrics

| Item | Detail |
|---|---|
| **Dataset** | 20,000+ synthetic customers |
| **Survival model** | Cox Proportional Hazards (`lifelines`) |
| **CLV model** | XGBoost Regressor |
| **Evaluation** | C-index · RMSE · MAE · R² |
| **Regions** | NA · EU · APAC · LATAM · MENA |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Survival Analysis** | `lifelines` |
| **ML Models** | `scikit-learn` · `XGBoost` |
| **Data Processing** | `pandas` · `numpy` |
| **Visualisation** | `plotly` · `matplotlib` · `seaborn` |
| **Dashboard** | `Dash` (Plotly) |

---

## 💡 Business Use Cases

- **Retention campaigns** — identify high-risk customers before they churn and intervene early
- **Revenue forecasting** — project next-quarter revenue using per-customer CLV estimates
- **Customer tiering** — prioritise high-CLV accounts for premium support
- **Marketing ROI** — compare CLV across acquisition channels to optimise spend

---

## 🔬 Key Features

- **Survival Analysis**: Kaplan-Meier curves for retention rates
- ** Cox Proportional Hazards**: Identifying risk factors (covariates) for churn
- **RFM Analysis**: Recency, Frequency, Monetary value feature engineering
- **CLV Prediction**: Regression models to forecast future revenue
- **Model Evaluation**: C-index (Concordance), RMSE, MAE

## 📂 Data Source

The dataset is **synthetic**, generated by `data/generate_clv_data.py` to mimic a subscription-based SaaS business:
- **Customers**: 20,000+ profiles with signup dates, regions (NA, EU, APAC, LATAM, MENA), and acquisition channels.
- **Transactions**: Transaction records linked to customers with realistic churn patterns.
- **patterns**: Simulates realistic churn decay curves and purchase consistency for accurate Lifetime Value (CLV) modeling.
