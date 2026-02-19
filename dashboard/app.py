"""
Interactive CLV & Churn Dashboard
=================================
Dash application for visualizing customer retention, churn risk factors,
and predicted lifetime value.
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.survival_analysis import SurvivalModel
from src.feature_engineering import FeatureEngineer
from src.regression_model import CLVRegressor, CLVVisualizer

# Initialize App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Customer Lifetime Value & Churn Dashboard"

# Load Data & Train Models (Cached for Simplicity)
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data')
try:
    customers = pd.read_csv(os.path.join(DATA_PATH, 'customers.csv'), parse_dates=['signup_date', 'churn_date'])
    transactions = pd.read_csv(os.path.join(DATA_PATH, 'transactions.csv'), parse_dates=['transaction_date'])
    
    # Train Survival Model
    survival = SurvivalModel()
    km_overall = survival.fit_kaplan_meier(customers['duration_days'], customers['churn_event'], 'Overall')
    
    # Train CLV Model (Simplified for Demo)
    fe = FeatureEngineer()
    cutoff_date = transactions['transaction_date'].max() - pd.Timedelta(days=365)
    X, y = fe.create_clv_dataset(transactions, customers, cutoff_date)
    regressor = CLVRegressor()
    clv_results = regressor.train(X, y)
    
    avg_clv = y.mean()
    churn_rate = customers['churn_event'].mean()
    
except Exception as e:
    print(f"Error loading data: {e}")
    customers = pd.DataFrame()
    avg_clv = 0
    churn_rate = 0

# Layout
navbar = dbc.NavbarSimple(
    brand="💎 Customer Lifetime Value (CLV) Predictor",
    brand_href="#",
    color="dark",
    dark=True,
)

metrics_row = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H4("Avg Lifetime Value", className="card-title"),
            html.H2(f"${avg_clv:,.2f}", className="text-success")
        ])
    ]), width=3),
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H4("Churn Rate", className="card-title"),
            html.H2(f"{churn_rate:.1%}", className="text-danger")
        ])
    ]), width=3),
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H4("Median Lifetime", className="card-title"),
            html.H2(f"{km_overall.get('median_survival_time', 0):.0f} Days", className="text-info")
        ])
    ]), width=3),
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H4("Analysis Status", className="card-title"),
            html.H2("Active", className="text-primary")
        ])
    ]), width=3),
], className="mb-4 mt-4")

tabs = dcc.Tabs([
    # Tab 1: Survival Analysis
    dcc.Tab(label='⏳ Retention & Churn', children=[
        dbc.Row([
            dbc.Col([
                html.Label("Compare Retention By:"),
                dcc.Dropdown(
                    id='group-selector',
                    options=[{'label': c, 'value': c} for c in ['plan_type', 'region', 'channel']],
                    value='plan_type'
                ),
            ], width=4, className="mt-3"),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='survival-curve'), width=12)
        ])
    ]),
    
    # Tab 2: CLV Prediction
    dcc.Tab(label='💰 Lifetime Value Prediction', children=[
        dbc.Row([
            dbc.Col(dcc.Graph(id='feature-importance', figure=CLVVisualizer.plot_feature_importance(clv_results['feature_importance'])), width=6),
            dbc.Col(dcc.Graph(id='actual-vs-pred', figure=CLVVisualizer.plot_actual_vs_predicted(clv_results['y_test'], clv_results['y_pred'])), width=6),
        ], className="mt-3")
    ]),
    
    # Tab 3: Customer Simulator
    dcc.Tab(label='🔮 Predict New Customer', children=[
        dbc.Row([
            dbc.Col([
                html.H4("Customer Attributes", className="mt-3"),
                dbc.Form([
                    html.Label("Plan Type"),
                    dcc.Dropdown(id='input-plan', options=[{'label': 'Basic', 'value': 'Basic'}, {'label': 'Pro', 'value': 'Pro'}, {'label': 'Enterprise', 'value': 'Enterprise'}], value='Pro'),
                    html.Br(),
                    html.Label("Region"),
                    dcc.Dropdown(id='input-region', options=[{'label': 'NA', 'value': 'NA'}, {'label': 'EU', 'value': 'EU'}, {'label': 'APAC', 'value': 'APAC'}], value='NA'),
                    html.Br(),
                    html.Label("Acquisition Channel"),
                    dcc.Dropdown(id='input-channel', options=[{'label': 'Organic', 'value': 'Organic'}, {'label': 'Ads', 'value': 'Ads'}, {'label': 'Referral', 'value': 'Referral'}], value='Ads'),
                    html.Br(),
                    html.Label("Days Since First Purchase (Tenure)"),
                    dbc.Input(id='input-tenure', type='number', value=30),
                    html.Br(),
                    html.Label("Avg Transaction Value"),
                    dbc.Input(id='input-monetary', type='number', value=50),
                    html.Br(),
                    html.Label("Frequency (Txns/Year)"),
                    dbc.Input(id='input-frequency', type='number', value=12),
                    html.Br(),
                    dbc.Button("Predict CLV", id='predict-btn', color="success", className="mt-3")
                ])
            ], width=4),
            dbc.Col([
                 html.H4("Prediction Result", className="mt-3"),
                 html.Div(id='prediction-output', className="p-5 border rounded bg-light text-center")
            ], width=8)
        ])
    ])
])

app.layout = dbc.Container([
    navbar,
    metrics_row,
    tabs
], fluid=True)

# Callbacks
@app.callback(
    Output('survival-curve', 'figure'),
    Input('group-selector', 'value')
)
def update_survival(group_col):
    fig = go.Figure()
    
    for group in customers[group_col].unique():
        mask = customers[group_col] == group
        if mask.sum() < 10: continue
            
        kmf = survival.fit_kaplan_meier(
            customers.loc[mask, 'duration_days'], 
            customers.loc[mask, 'churn_event'], 
            str(group)
        )
        
        fig.add_trace(go.Scatter(
            x=kmf['timeline'], y=kmf['survival_function'],
            mode='lines', name=str(group)
        ))
    
    fig.update_layout(
        title=f"retention Curve by {group_col}",
        xaxis_title="Days Since Signup",
        yaxis_title="Retention Rate",
        template='plotly_white'
    )
    return fig

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('input-plan', 'value'),
    State('input-region', 'value'),
    State('input-channel', 'value'),
    State('input-tenure', 'value'),
    State('input-monetary', 'value'),
    State('input-frequency', 'value')
)
def predict_new_customer(n_clicks, plan, region, channel, tenure, monetary, frequency):
    if not n_clicks:
        return "Enter details and click Predict."
    
    # Prepare single-row DataFrame (matching training features)
    input_data = pd.DataFrame([{
        'recency': 30, # Assumed active
        'frequency': int(frequency),
        'monetary': float(monetary) * int(frequency), # Total spend approx
        'avg_order_value': float(monetary),
        'tenure_days': int(tenure),
        'purchases_per_month': int(frequency) / 12,
        'is_active': True,
        'plan_type': plan,
        'region': region,
        'channel': channel
    }])
    
    # Predict
    pred_val = regressor.predict(input_data)[0]
    
    return html.div([
        html.H3(f"Predicted Future Value (1 Year)", className="text-secondary"),
        html.H1(f"${pred_val:,.2f}", className="text-success display-4"),
        html.P(f"Based on {plan} plan customer in {region} region.", className="mt-3")
    ])

if __name__ == '__main__':
    app.run(debug=True, port=8051)
