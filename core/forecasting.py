"""
RAG-Driven Analytics: Forecasting Engine
Implements time-series analysis for asset price predictions utilizing statistical modeling.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go

class RealEstatePredictor:
    """Predictive modeling for real estate market trends."""
    
    def __init__(self):
        self.model = None
        self.historical_data = None
        self.region_name = ""

    def prepare_data(self, df: pd.DataFrame, region: str) -> pd.DataFrame:
        """Transforms raw Zillow data into Prophet-ready format."""
        try:
            date_columns = [col for col in df.columns if str(col).startswith('20')]
            
            df_melted = pd.melt(
                df,
                id_vars=['RegionName', 'RegionID', 'StateName'],
                value_vars=date_columns,
                var_name='ds',
                value_name='y'
            )
            
            df_melted['ds'] = pd.to_datetime(df_melted['ds'])
            df_region = df_melted[df_melted['RegionName'] == region].copy()
            df_region = df_region.sort_values('ds').dropna(subset=['y'])
            
            self.historical_data = df_region[['ds', 'y']].copy()
            self.region_name = region
            
            return self.historical_data
            
        except Exception as e:
            raise ValueError(f"Data Preparation Failure: {str(e)}")
    
    def train(self, df: pd.DataFrame, region: str):
        """Trains the Prophet model on historical data."""
        data = self.prepare_data(df, region)
        
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )
        
        self.model.fit(data)
        return True

    def evaluate_performance(self):
        """Calculates statistical metrics for model validation."""
        if not self.model or self.historical_data is None:
            return None
            
        forecast = self.model.predict(self.historical_data[['ds']])
        actual = self.historical_data['y'].values
        predicted = forecast['yhat'].values
        
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        return {
            'MAPE': f"{mape:.2f}%",
            'RMSE': f"${rmse:,.2f}",
            'Confidence': f"{100 - mape:.2f}%"
        }
    
    def generate_forecast(self, months: int):
        """Generates future predictions and interactive visualizations."""
        if not self.model:
            return None, None
            
        future = self.model.make_future_dataframe(periods=months, freq='ME')
        forecast = self.model.predict(future)
        
        # Professional Visualization
        fig = go.Figure()
        
        # Historical Baseline
        fig.add_trace(go.Scatter(
            x=self.historical_data['ds'],
            y=self.historical_data['y'],
            mode='lines',
            name='Historical Data',
            line=dict(color='#2E86C1', width=2)
        ))
        
        # Predicted Trend
        future_forecast = forecast.tail(months)
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat'],
            mode='lines',
            name='Forecasted Trend',
            line=dict(color='#E74C3C', width=3, dash='dash')
        ))
        
        # Confidence Corridor
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat_lower'],
            fill='tonexty',
            fillcolor='rgba(231, 76, 60, 0.2)',
            mode='lines',
            line=dict(width=0),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title=f'Market Analysis: {self.region_name}',
            xaxis_title='Timeline',
            yaxis_title='Valuation ($)',
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], fig
