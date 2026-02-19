"""
Feature Engineering Module
==========================
Transforms raw transaction logs into RFM features and user-level attributes
for CLV prediction models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List

class FeatureEngineer:
    """
    Computes Recency, Frequency, Monetary (RFM) and behavioral features.
    """
    
    def __init__(self, observation_window: int = 365, prediction_window: int = 365):
        self.obs_window = observation_window
        self.pred_window = prediction_window
        
    def transform_transactions(
        self,
        transactions: pd.DataFrame,
        customers: pd.DataFrame,
        reference_date: datetime = None
    ) -> pd.DataFrame:
        """
        Aggregate transactions to customer level.
        Calculates RFM metrics and churn status.
        """
        if reference_date is None:
            reference_date = transactions['transaction_date'].max()
            
        # Merge customers with transactions
        df = transactions.merge(customers[['customer_id', 'signup_date', 'plan_type', 'region', 'channel']], on='customer_id', how='left')
        
        # Calculate RFM
        rfm = df.groupby('customer_id').agg(
            recency=('transaction_date', lambda x: (reference_date - x.max()).days),
            frequency=('transaction_date', 'count'),
            monetary=('amount', 'sum'),
            avg_order_value=('amount', 'mean'),
            first_purchase=('transaction_date', 'min'),
            last_purchase=('transaction_date', 'max'),
            tenure_days=('signup_date', lambda x: (reference_date - x.iloc[0]).days)
        ).reset_index()
        
        # Add Demographics (first value per group)
        demographics = df.groupby('customer_id')[['plan_type', 'region', 'channel']].first().reset_index()
        rfm = rfm.merge(demographics, on='customer_id', how='left')
        
        # Derived Features
        rfm['purchases_per_month'] = rfm['frequency'] / (rfm['tenure_days'] / 30).clip(lower=1)
        rfm['is_active'] = rfm['recency'] <= 90  # Simple definition: active if purchased in last 90 days
        
        return rfm

    def create_clv_dataset(
        self,
        transactions: pd.DataFrame,
        customers: pd.DataFrame,
        split_date: datetime
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create X (historical features) and y (future value) for supervised learning.
        
        - X: Features from signup up to split_date
        - y: Revenue from split_date to split_date + prediction_window
        """
        # Filter transactions
        historical = transactions[transactions['transaction_date'] <= split_date]
        future = transactions[(transactions['transaction_date'] > split_date) & 
                              (transactions['transaction_date'] <= split_date + timedelta(days=self.pred_window))]
        
        # Calculate Features (X)
        X = self.transform_transactions(historical, customers, reference_date=split_date)
        
        # Calculate Target (y)
        y = future.groupby('customer_id')['amount'].sum().reset_index()
        y.rename(columns={'amount': 'future_clv'}, inplace=True)
        
        # Merge target into X (filling 0 for users with no future revenue)
        data = X.merge(y, on='customer_id', how='left')
        data['future_clv'].fillna(0, inplace=True)
        
        return data.drop(columns=['future_clv']), data['future_clv']

