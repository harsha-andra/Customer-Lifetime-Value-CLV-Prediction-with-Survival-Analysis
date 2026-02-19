"""
CLV & Survival Analysis Data Generator
======================================
Generates customer transaction logs and subscription lifecycle data
suitable for Kaplan-Meier, Cox PH, and CLV regression models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generate_customer_lifecycle(n_users: int = 20000) -> pd.DataFrame:
    """
    Generate customer lifecycle data including signup, churn dates,
    and demographics (plan type, region, acquisition channel).
    """
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    # User IDs
    user_ids = np.arange(1, n_users + 1)
    
    # Signup Dates (uniform over 2 years)
    delta_days = (end_date - start_date).days
    signup_offsets = np.random.randint(0, delta_days, n_users)
    signup_dates = [start_date + timedelta(days=int(d)) for d in signup_offsets]
    
    # Demographics
    plans = np.random.choice(['Basic', 'Pro', 'Enterprise'], size=n_users, p=[0.6, 0.3, 0.1])
    regions = np.random.choice(['NA', 'EU', 'APAC', 'LATAM', 'MENA', 'SA'], size=n_users, p=[0.35, 0.25, 0.20, 0.10, 0.05, 0.05])
    channels = np.random.choice(['Organic', 'Ads', 'Referral', 'Affiliate', 'Social', 'Email'], size=n_users, p=[0.25, 0.30, 0.15, 0.10, 0.10, 0.10])
    
    # Churn Logic: Depends on Plan & Channel (Cox covariate effect)
    # Base hazard rate (lambda) for exponential distribution
    base_lambda = 1/365  # Avg lifetime ~1 year
    
    # Adjust lambda based on features
    # Pro plans churn less, Referrals churn less
    multipliers = []
    for p, c in zip(plans, channels):
        m = 1.0
        if p == 'Pro': m *= 0.7
        elif p == 'Enterprise': m *= 0.4
        
        if c == 'Referral': m *= 0.6
        elif c == 'Ads': m *= 1.3
        
        multipliers.append(m)
    
    adjusted_lambdas = base_lambda * np.array(multipliers)
    lifetimes = np.random.exponential(1 / adjusted_lambdas)
    
    # Calculate Churn Date
    churn_dates = []
    censored = []  # 0 = Churned (Uncensored), 1 = Active (Censored)
    
    for signup, lifetime in zip(signup_dates, lifetimes):
        churn_dt = signup + timedelta(days=int(lifetime))
        if churn_dt > end_date:
            churn_dates.append(end_date)
            censored.append(1)  # Still active at end of observation
        else:
            churn_dates.append(churn_dt)
            censored.append(0)  # Churned
            
    # Calculate duration (T) and Event (E)
    duration = [(c - s).days for c, s in zip(churn_dates, signup_dates)]
    event = [1 if c == 0 else 0 for c in censored] # Event = Churn (1)
    
    customers = pd.DataFrame({
        'customer_id': user_ids,
        'signup_date': signup_dates,
        'churn_date': churn_dates,
        'duration_days': duration,
        'churn_event': event,  # 1 if churned, 0 if censored
        'plan_type': plans,
        'region': regions,
        'channel': channels
    })
    
    return customers


def generate_transactions(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Generate transaction history for each customer based on their
    active duration and plan type.
    """
    transactions = []
    
    for _, row in customers.iterrows():
        cust_id = row['customer_id']
        start = row['signup_date']
        end = row['churn_date']
        plan = row['plan_type']
        
        # Transaction frequency (purchase every X days)
        if plan == 'Basic':
            freq = 30  # Monthly sub
            value_mean = 9.99
            value_std = 0
        elif plan == 'Pro':
            freq = 30
            value_mean = 29.99
            value_std = 5.00 # Upsells
        else: # Enterprise
            freq = 365 # Yearly
            value_mean = 999.00
            value_std = 200.00
            
        current_date = start
        while current_date <= end:
            # Random variation in purchase date
            txn_date = current_date + timedelta(days=np.random.randint(-2, 3))
            if txn_date > end:
                break
                
            amount = max(0, np.random.normal(value_mean, value_std))
            
            transactions.append({
                'customer_id': cust_id,
                'transaction_date': txn_date,
                'amount': round(amount, 2)
            })
            
            # Next transaction
            next_days = int(np.random.normal(freq, 2))
            current_date += timedelta(days=max(1, next_days))
            
    return pd.DataFrame(transactions)


if __name__ == '__main__':
    print("Generating Customer & Transaction Data...")
    
    customers = generate_customer_lifecycle(20000)
    customers.to_csv(os.path.join(os.path.dirname(__file__), 'customers.csv'), index=False)
    print(f"  ✅ customers.csv — {len(customers):,} users")
    
    transactions = generate_transactions(customers)
    transactions.to_csv(os.path.join(os.path.dirname(__file__), 'transactions.csv'), index=False)
    print(f"  ✅ transactions.csv — {len(transactions):,} records")
