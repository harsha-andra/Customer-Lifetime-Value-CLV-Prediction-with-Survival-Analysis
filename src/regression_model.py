"""
CLV Regression & Visualization Module
=====================================
Combines XGBoost regression for lifetime value prediction and visualization
utilities for model performance evaluation.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Tuple

class CLVRegressor:
    """
    Predicts future Customer Lifetime Value (CLV) using XGBoost.
    """
    
    def __init__(self, objective: str = 'reg:squarederror'):
        self.model = xgb.XGBRegressor(objective=objective, n_estimators=100, learning_rate=0.1)
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train XGBoost model on historical features to predict future value.
        """
        # Preprocessing: drop non-numeric or leak columns
        X_train = X.drop(columns=['customer_id', 'signup_date', 'first_purchase', 'last_purchase'], 
                         errors='ignore')
        
        # One-hot encoding for categorical features
        X_encoded = pd.get_dummies(X_train, columns=['plan_type', 'region', 'channel'], drop_first=True)
        
        # Train-Test Split
        X_train_split, X_test_split, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train_split, y_train)
        
        # Evaluate
        preds = self.model.predict(X_test_split)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        return {
            'mae': round(mae, 2),
            'r2_score': round(r2, 4),
            'feature_importance': dict(zip(X_encoded.columns, self.model.feature_importances_)),
            'y_test': y_test.tolist(),
            'y_pred': preds.tolist()
        }

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        # Similar preprocessing (simplified for demo)
        X_encoded = pd.get_dummies(X_new.drop(columns=['customer_id'], errors='ignore'), 
                                   columns=['plan_type', 'region', 'channel'], drop_first=True)
        # Ensure columns match training (simplified: use reindex)
        return self.model.predict(X_encoded)


class CLVVisualizer:
    """
    Visualization tools for Survival Analysis and CLV Prediction.
    """
    
    @staticmethod
    def plot_survival_curve(km_result: Dict, label: str = 'Survival Probability') -> go.Figure:
        """
        Plot Kaplan-Meier survival curve with confidence intervals using Plotly.
        """
        timeline = km_result['timeline']
        survival = km_result['survival_function']
        ci_lower = km_result['confidence_interval'][f'{label}_lower_0.95']
        ci_upper = km_result['confidence_interval'][f'{label}_upper_0.95']
        
        fig = go.Figure()
        
        # Survival Line
        fig.add_trace(go.Scatter(x=timeline, y=survival, mode='lines', name=label,
                                 line=dict(color='blue', width=2)))
        
        # Confidence Interval (shaded area)
        fig.add_trace(go.Scatter(
            x=timeline + timeline[::-1],
            y=ci_upper + ci_lower[::-1],
            fill='toself',
            fillcolor='rgba(0,100,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))
        
        fig.update_layout(title="Customer Survival Curve (Kaplan-Meier)",
                          xaxis_title="Days Since Signup",
                          yaxis_title="Survival Probability (Retention)",
                          template='plotly_white')
        return fig

    @staticmethod
    def plot_feature_importance(importance_dict: Dict) -> go.Figure:
        """
        Bar chart of XGBoost feature importance.
        """
        df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
        df = df.sort_values('Importance', ascending=True)
        
        fig = px.bar(df, x='Importance', y='Feature', orientation='h',
                     title="Key Drivers of Customer Lifetime Value (Feature Importance)")
        fig.update_layout(template='plotly_white')
        return fig

    @staticmethod
    def plot_actual_vs_predicted(y_true: list, y_pred: list) -> go.Figure:
        """
        Scatter plot of Actual vs. Predicted CLV.
        """
        fig = px.scatter(x=y_true, y=y_pred, opacity=0.5,
                         labels={'x': 'Actual Future Value ($)', 'y': 'Predicted Future Value ($)'},
                         title="Model Performance: Actual vs Predicted CLV")
        
        # Add diagonal line
        max_val = max(max(y_true), max(y_pred))
        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(color="red", dash="dash"))
        
        fig.update_layout(template='plotly_white')
        return fig
