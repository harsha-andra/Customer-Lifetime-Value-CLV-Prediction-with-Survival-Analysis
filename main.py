"""
CLV Prediction & Lifecycle Analysis Pipeline
============================================
Runs the full customer analysis pipeline: data loading, survival analysis,
RFM feature engineering, and CLV prediction model training.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from src.survival_analysis import SurvivalModel
from src.feature_engineering import FeatureEngineer
from src.regression_model import CLVRegressor

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super(NpEncoder, self).default(obj)

# Constants
OBSERVATION_WINDOW = 365
PREDICTION_WINDOW = 365
SPLIT_DATE = datetime(2024, 12, 31)

def run_pipeline():
    print("🚀 Starting Customer Lifetime Value (CLV) Analysis Pipeline...")
    
    # 1. Load Data
    try:
        cust_path = os.path.join('data', 'customers.csv')
        txn_path = os.path.join('data', 'transactions.csv')
        
        customers = pd.read_csv(cust_path, parse_dates=['signup_date', 'churn_date'])
        transactions = pd.read_csv(txn_path, parse_dates=['transaction_date'])
        
        print(f"\n✅ Loaded {len(customers):,} customers and {len(transactions):,} transactions.")
    except FileNotFoundError:
        print("❌ Data not found. Please run 'python data/generate_clv_data.py' first.")
        return

    # 2. Survival Analysis (Kaplan-Meier & Cox PH)
    print("\n⏳ 1. Survival Analysis (Retention Modeling)")
    survival = SurvivalModel()
    
    # Kaplan-Meier Curve
    km_result = survival.fit_kaplan_meier(
        durations=customers['duration_days'],
        event_observed=customers['churn_event'],
        label='Overall Retention'
    )
    print(f"   - Median Survival Time: {km_result['median_survival_time']:.1f} days")
    print(f"   - Retention at 1 Year: {km_result['survival_probability_at_1y']:.1%}")
    
    # Compare Groups (Plan Type)
    log_rank = survival.compare_survival_curves(customers, 'plan_type')
    if 'log_rank_test' in log_rank:
        lr = log_rank['log_rank_test']
        print(f"   - Log-Rank Test ({lr['group_A']} vs {lr['group_B']}): p={lr['p_value']:.4f} {'✅ Significant' if lr['significant'] else '❌ Not Significant'}")
    
    # Cox Proportional Hazards (Risk Factors)
    cox_results = survival.fit_cox_ph(
        customers, 
        covariates=['plan_type', 'region', 'channel'] # Categoricals handled inside
    )
    print(f"   - Cox Model Concordance Index: {cox_results['concordance_index']:.4f}")
    print(f"   - Top Hazard Ratios (Risk Multipliers):")
    for feat, stats in sorted(cox_results['hazard_ratios'].items(), key=lambda x: x[1]['hazard_ratio'], reverse=True)[:3]:
        print(f"     • {feat}: {stats['hazard_ratio']:.2f}x risk (p={stats['p_value']:.4f})")

    # 3. Feature Engineering (RFM)
    print("\n📊 2. RFM Feature Engineering")
    fe = FeatureEngineer(prediction_window=365)
    
    # Create dataset for CLV prediction (split at cutoff date)
    # Train on history up to 2024-12-31, predict revenue for 2025
    X_train, y_train = fe.create_clv_dataset(transactions, customers, SPLIT_DATE)
    
    print(f"   - Generated {len(X_train):,} training examples.")
    print(f"   - Average Future Value (Target): ${y_train.mean():.2f}")

    # 4. CLV Regression Model (XGBoost)
    print("\n🤖 3. CLV Prediction (XGBoost Regressor)")
    regressor = CLVRegressor()
    model_results = regressor.train(X_train, y_train)
    
    print(f"   - Model Performance (R²): {model_results['r2_score']:.4f}")
    print(f"   - Mean Absolute Error (MAE): ${model_results['mae']:.2f}")
    
    # 5. Save Summary Report
    os.makedirs('reports', exist_ok=True)
    report_path = os.path.join('reports', 'analysis_summary.json')
    
    summary = {
        'survival': {
            'median_days': km_result['median_survival_time'],
            'retention_1y': km_result['survival_probability_at_1y'],
            'cox_c_index': cox_results['concordance_index']
        },
        'clv_model': {
            'r2': model_results['r2_score'],
            'mae': model_results['mae'],
            'top_features': dict(sorted(model_results['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5])
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(summary, f, cls=NpEncoder, indent=4)
    print(f"\n📄 Saved analysis summary to {report_path}")

if __name__ == '__main__':
    run_pipeline()
