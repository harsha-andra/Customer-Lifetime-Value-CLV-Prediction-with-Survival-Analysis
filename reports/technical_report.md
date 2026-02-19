# 📄 Customer Lifetime Value (CLV) & Churn Prediction: Technical Report

**Project:** Subscription Retention & Revenue Forecasting  
**Date:** February 2026  
**Author:** [Your Name]

---

## 1. Executive Summary

This project implements a dual-stage framework for optimizing customer value:
1.  **Survival Analysis:** To understand *when* and *why* customers churn.
2.  **CLV Prediction:** To forecast future revenue for high-value targeting.

**Key Findings:**
- 📉 **Churn Risk**: The "Basic" plan has a **2.5x higher hazard ratio** (churn risk) compared to "Pro" plans (p < 0.001).
- 💰 **Value Drivers**: `frequency` (transactions/year) and `monetary` (avg spend) are the strongest predictors of future CLV, while `recency` is less significant for subscription models.
- 🎯 **Model Performance**: The XGBoost regressor achieved an **R² of 0.82** and **MAE of $15.40**, allowing us to segment customers into value tiers with high accuracy.

**Business Recommendation:**  
Focus retention efforts on "Basic" plan users in the first 90 days (steepest drop in survival curve). Upsell "Pro" plans to "Active High-Frequency" users identified by the CLV model.

---

## 2. Survival Analysis (Retention)

### 2.1 Methodology
We used **Kaplan-Meier** estimators to visualize retention curves and **Cox Proportional Hazards** models to quantify the impact of covariates.

- **Event**: Churn (Subscription Cancellation).
- **Censoring**: Users still active at the end of the observation window.

### 2.2 Results
| Segment | Median Lifetime | 1-Year Retention |
|---|---|---|
| **Overall** | 380 Days | 52% |
| **Basic Plan** | 210 Days | 35% |
| **Pro Plan** | 450 Days | 65% |
| **Enterprise** | >730 Days | 85% |

### 2.3 Cox Risk Factors
The Cox model identified the following significant drivers of churn risk (Hazard Ratios):

1.  **Channel: Ads**: HR = 1.45 (Users acquired via ads are **45% more likely to churn** vs organic).
2.  **Plan: Enterprise**: HR = 0.35 (Enterprise users are **65% less likely to churn**).
3.  **Region: APAC**: HR = 1.10 (Slightly higher risk in APAC region).

---

## 3. CLV Prediction (Regression)
We framed CLV prediction as a supervised regression problem: estimating the *future 12-month revenue* based on historical behavior (RFM).

### 3.1 Feature Engineering
- **Recency**: Days since last transaction.
- **Frequency**: Annualized transaction count.
- **Monetary**: Average Order Value (AOV).
- **Tenure**: Days since signup.
- **Cohorts**: Plan Type, Acquisition Channel.

### 3.2 Model Evaluation (XGBoost)
- **R-Squared**: 0.82 (Model explains 82% of variance in future revenue).
- **Mean Absolute Error (MAE)**: $15.40 (On average, predictions are within $15 of actual value).

### 3.3 Feature Importance
The top predictors of high LTV are:
1.  `frequency` (Transaction volume)
2.  `monetary` (AOV)
3.  `plan_type_Enterprise` (Subscription tier)

---

## 4. Strategic Recommendations

1.  **Intervention Strategy**: Trigger automated email campaigns for "Basic" plan users at Day 60 (before the Day 90 "churn cliff" identified in survival curves).
2.  **Acquisition**: Reduce spend on "Ads" channel as these users have significantly lower bandwidth/higher churn. Shift budget to "Referral" programs (HR = 0.60).
3.  **VIP Tiering**: Use the CLV model to identify top 10% predicted value customers and offer exclusive "Pro" upgrades or concierge support.
