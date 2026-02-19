"""
Survival Analysis Module
========================
Implements Kaplan-Meier survival curves and Cox Proportional Hazards regression
for customer churn prediction.
"""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class SurvivalModel:
    """
    Wrapper for lifelines survival analysis models.
    """
    
    def __init__(self):
        self.kmf = KaplanMeierFitter()
        self.cph = CoxPHFitter()
        
    def fit_kaplan_meier(
        self,
        durations: pd.Series,
        event_observed: pd.Series,
        label: str = 'Overall'
    ) -> Dict:
        """
        Fit Kaplan-Meier survival curve.
        Returns survival function and median survival time.
        """
        self.kmf.fit(durations, event_observed, label=label)
        
        return {
            'median_survival_time': self.kmf.median_survival_time_,
            'survival_probability_at_1y': self.kmf.predict(365),
            'survival_probability_at_2y': self.kmf.predict(730),
            'timeline': self.kmf.timeline.tolist(),
            'survival_function': self.kmf.survival_function_[label].tolist(),
            'confidence_interval': self.kmf.confidence_interval_.to_dict()
        }
    
    def compare_survival_curves(
        self,
        df: pd.DataFrame,
        group_col: str,
        duration_col: str = 'duration_days',
        event_col: str = 'churn_event'
    ) -> Dict:
        """
        Perform Log-Rank test to compare survival curves between groups.
        """
        groups = df[group_col].unique()
        results = {}
        
        # Pairwise comparison (simplified to first 2 groups for demo)
        if len(groups) >= 2:
            g1 = groups[0]
            g2 = groups[1]
            
            T1 = df[df[group_col] == g1][duration_col]
            E1 = df[df[group_col] == g1][event_col]
            T2 = df[df[group_col] == g2][duration_col]
            E2 = df[df[group_col] == g2][event_col]
            
            lr_result = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
            
            results['log_rank_test'] = {
                'group_A': str(g1),
                'group_B': str(g2),
                'test_statistic': round(lr_result.test_statistic, 4),
                'p_value': round(lr_result.p_value, 6),
                'significant': lr_result.p_value < 0.05
            }
            
        return results

    def fit_cox_ph(
        self,
        df: pd.DataFrame,
        duration_col: str = 'duration_days',
        event_col: str = 'churn_event',
        covariates: List[str] = None
    ) -> Dict:
        """
        Fit Cox Proportional Hazards model to estimate covariate effects.
        """
        if covariates:
            # One-hot encode categorical variables
            df_model = pd.get_dummies(df[[duration_col, event_col] + covariates], drop_first=True)
        else:
            df_model = df[[duration_col, event_col]]
            
        self.cph.fit(df_model, duration_col=duration_col, event_col=event_col)
        
        # Extract hazard ratios
        summary = self.cph.summary[['coef', 'exp(coef)', 'p']]
        summary.columns = ['log_hazard_ratio', 'hazard_ratio', 'p_value']
        
        return {
            'concordance_index': round(self.cph.concordance_index_, 4),
            'hazard_ratios': summary.to_dict(orient='index'),
            'AIC': round(self.cph.AIC_partial_, 2),
            'log_likelihood': round(self.cph.log_likelihood_, 2)
        }
    
    def predict_survival_probability(
        self,
        covariates_df: pd.DataFrame,
        times: List[int]
    ) -> pd.DataFrame:
        """
        Predict survival probability at specific time points for new data.
        """
        # Ensure dummy columns match trained model
        # (Simplified: assumes input df matches encoded structure)
        return self.cph.predict_survival_function(covariates_df, times=times)

