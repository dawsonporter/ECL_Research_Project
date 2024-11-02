import warnings
import requests
import pandas as pd
import numpy as np
from typing import List, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from scipy import stats

BASE_URL = "https://banks.data.fdic.gov/api"

class ECLMetrics:
    def __init__(self):
        self.raw_metrics = {
            'allowance_metrics': [
                'LNATRES',  # Allowance for Credit Loss
                'P3ASSET',  # Past Due 30-89 Days
                'P9ASSET',  # Past Due 90+ Days
                'DRLNLS',   # Total Charge-Offs
                'CRLNLS',   # Total Recoveries
                'NTLNLSQ'   # Quarterly Net Charge-Offs
            ],
            'loan_metrics': [
                'LNLSGR',   # Total Loans and Leases
                'LNRE',     # Real Estate Loans
                'LNRECONS', # Construction & Land Development
                'LNRENRES', # Non-residential Real Estate
                'LNCI',     # Commercial & Industrial Loans
                'LNCONOTH', # Consumer Loans
                'LNRERES',  # Residential Real Estate Loans
                'LNREMULT', # Multifamily Loans
                'LNATRES'   # Allowance for Credit Loss
            ],
            'capital_metrics': [
                'RBCT1J',   # Tier 1 Capital
                'RBC1AAJ',  # Leverage Ratio
                'RBCRWAJ',  # Risk-Based Capital Ratio
                'EQ',       # Total Equity
                'ASSET'     # Total Assets
            ],
            'performance_metrics': [
                'NETINC',   # Net Income
                'ROA',      # Return on Assets
                'ROE',      # Return on Equity
                'NIMY',     # Net Interest Margin
                'EEFFR'     # Efficiency Ratio
            ]
        }
        
        self.stress_periods = {
            'gfc': {
                'start': '20080101',
                'end': '20101231',
                'name': 'Global Financial Crisis'
            },
            'covid': {
                'start': '20200101',
                'end': '20211231',
                'name': 'COVID-19 Crisis'
            }
        }
        
        # Define required columns for each hypothesis
        self.required_columns = {
            'h1': ['lnrecons', 'lnrenres', 'lnlsgr', 'lnatres'],
            'h2': ['lnconoth', 'lnlsgr', 'lnatres'],
            'h3': ['rbct1j', 'lnlsgr', 'lnatres'],
            'h4': ['lnatres', 'lnlsgr', 'netinc']
        }

class ECLDataProcessor:
    def __init__(self):
        self.metrics = ECLMetrics()
        
    def process_raw_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """Convert raw API data into structured DataFrame"""
        if not raw_data:
            return pd.DataFrame()
            
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(raw_data)
        
        # Convert column names to lowercase
        df.rename(columns=str.lower, inplace=True)
        
        # Convert date format
        if 'repdte' in df.columns:
            df['date'] = pd.to_datetime(df['repdte'], format='%Y%m%d')
        
        # Convert numeric columns
        non_numeric_cols = ['cert', 'repdte', 'date', 'bank', 'abbreviated_name']
        numeric_columns = [col for col in df.columns if col not in non_numeric_cols]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def calculate_ecl_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate core ECL metrics"""
        df = df.copy()
        
        try:
            # Calculate key ratios only if required columns exist
            if self.validate_required_columns(df, ['lnrecons', 'lnrenres', 'lnlsgr']):
                df['cre_concentration'] = (
                    (df['lnrecons'].fillna(0) + df['lnrenres'].fillna(0)) / 
                    df['lnlsgr'] * 100
                )
                
            if self.validate_required_columns(df, ['lnconoth', 'lnlsgr']):
                df['consumer_loan_ratio'] = (
                    df['lnconoth'].fillna(0) / df['lnlsgr'] * 100
                )
                
            if self.validate_required_columns(df, ['lnatres', 'lnlsgr']):
                df['ecl_coverage'] = (
                    df['lnatres'].fillna(0) / df['lnlsgr'] * 100
                )
                
            if self.validate_required_columns(df, ['ntlnlsq', 'lnlsgr']):
                df['nco_rate'] = (
                    (df['ntlnlsq'].fillna(0) * 4) / df['lnlsgr'] * 100
                )
                
            if self.validate_required_columns(df, ['rbct1j', 'lnlsgr']):
                df['tier1_ratio'] = (
                    df['rbct1j'].fillna(0) / df['lnlsgr'] * 100
                )
                
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            
        return df

    def validate_required_columns(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Check if DataFrame contains required columns"""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False
        return True

    def calculate_stress_period_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metric changes during stress periods"""
        df = df.copy()
        
        for period_name, period_info in self.metrics.stress_periods.items():
            period_mask = (
                (df['date'] >= pd.to_datetime(period_info['start'])) & 
                (df['date'] <= pd.to_datetime(period_info['end']))
            )
            
            # Calculate changes for each metric during stress period
            for metric in ['ecl_coverage', 'nco_rate', 'tier1_ratio']:
                if metric in df.columns:
                    change_col = f'{metric}_change_{period_name}'
                    df.loc[period_mask, change_col] = (
                        df.loc[period_mask, metric].groupby(df['cert']).pct_change() * 100
                    )
        
        return df

    def calculate_predicted_ecl_and_error(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate predicted ECL and prediction error"""
        df = df.copy()
        try:
            # Ensure necessary columns are present
            if self.validate_required_columns(df, ['lnlsgr', 'nco_rate', 'tier1_ratio', 'lnatres']):
                df['predicted_ecl'] = (
                    df['lnlsgr'] * 
                    (df['nco_rate'] / 100) * 
                    (1 + df['tier1_ratio'] / 100)
                )
                df['prediction_error'] = (
                    (df['predicted_ecl'] - df['lnatres']) / df['lnatres'] * 100
                )
        except Exception as e:
            print(f"Error calculating predicted ECL and error: {str(e)}")
        return df

class HypothesisTester:
    def __init__(self):
        self.metrics = ECLMetrics()
        self.significance_level = 0.05

    def test_hypothesis_1(self, df: pd.DataFrame, period: str) -> Dict:
        """Test if banks with high CRE concentration experience larger ECL increases"""
        try:
            period_info = self.metrics.stress_periods[period]
            
            # Filter for stress period
            stress_df = df[
                (df['date'] >= pd.to_datetime(period_info['start'])) &
                (df['date'] <= pd.to_datetime(period_info['end']))
            ].copy()
            
            # Remove any invalid values
            stress_df = stress_df.replace([np.inf, -np.inf], np.nan).dropna(
                subset=['cre_concentration', f'ecl_coverage_change_{period}']
            )
            
            # Create quartiles based on CRE concentration
            stress_df['cre_quartile'] = pd.qcut(
                stress_df['cre_concentration'], 
                q=4, 
                labels=['Q1', 'Q2', 'Q3', 'Q4']
            )
            
            # Calculate differences between top and bottom quartiles
            top_quartile = stress_df[stress_df['cre_quartile'] == 'Q4'][f'ecl_coverage_change_{period}']
            bottom_quartile = stress_df[stress_df['cre_quartile'] == 'Q1'][f'ecl_coverage_change_{period}']
            
            difference_pct = ((top_quartile.mean() - bottom_quartile.mean()) / 
                            abs(bottom_quartile.mean())) * 100
            
            # Perform statistical test
            stat, p_value = stats.mannwhitneyu(
                top_quartile.dropna(),
                bottom_quartile.dropna(),
                alternative='greater'
            )
            
            return {
                'difference_pct': difference_pct,
                'hypothesis_supported': difference_pct >= 20 and p_value < self.significance_level,
                'p_value': p_value,
                'statistic': stat,
                'top_quartile_mean': top_quartile.mean(),
                'bottom_quartile_mean': bottom_quartile.mean(),
                'sample_size': len(top_quartile) + len(bottom_quartile),
                'quartile_details': {
                    'Q4': {
                        'mean': top_quartile.mean(),
                        'median': top_quartile.median(),
                        'std': top_quartile.std(),
                        'count': len(top_quartile)
                    },
                    'Q1': {
                        'mean': bottom_quartile.mean(),
                        'median': bottom_quartile.median(),
                        'std': bottom_quartile.std(),
                        'count': len(bottom_quartile)
                    }
                }
            }
            
        except Exception as e:
            print(f"Error testing hypothesis 1: {str(e)}")
            return {}

    def test_hypothesis_2(self, df: pd.DataFrame, period: str) -> Dict:
        """Test consumer loan portfolio sensitivity during stress periods"""
        try:
            period_info = self.metrics.stress_periods[period]
            
            # Filter for stress period
            stress_df = df[
                (df['date'] >= pd.to_datetime(period_info['start'])) &
                (df['date'] <= pd.to_datetime(period_info['end']))
            ].copy()
            
            # Remove invalid values
            stress_df = stress_df.replace([np.inf, -np.inf], np.nan).dropna(
                subset=['consumer_loan_ratio', f'ecl_coverage_change_{period}']
            )

            # Calculate median consumer loan ratio
            median_consumer_ratio = stress_df['consumer_loan_ratio'].median()

            # Split banks by consumer loan concentration
            high_consumer = stress_df[stress_df['consumer_loan_ratio'] > median_consumer_ratio]
            low_consumer = stress_df[stress_df['consumer_loan_ratio'] <= median_consumer_ratio]

            # Calculate ECL changes
            metric = f'ecl_coverage_change_{period}'

            high_changes = high_consumer[metric].dropna()
            low_changes = low_consumer[metric].dropna()

            if len(high_changes) == 0 or len(low_changes) == 0:
                raise ValueError("No data available for high or low consumer loan groups.")

            # Perform statistical test
            stat, p_value = stats.mannwhitneyu(
                high_changes,
                low_changes,
                alternative='greater'
            )

            difference_pct = ((high_changes.mean() - low_changes.mean()) / 
                            abs(low_changes.mean())) * 100

            return {
                'difference_pct': difference_pct,
                'hypothesis_supported': difference_pct >= 0 and p_value < self.significance_level,
                'p_value': p_value,
                'statistic': stat,
                'high_consumer_mean': high_changes.mean(),
                'low_consumer_mean': low_changes.mean(),
                'median_consumer_ratio': median_consumer_ratio,
                'sample_size': {
                    'high_consumer': len(high_changes),
                    'low_consumer': len(low_changes)
                }
            }
            
        except Exception as e:
            print(f"Error testing hypothesis 2: {str(e)}")
            return {}

    def test_hypothesis_3(self, df: pd.DataFrame, period: str) -> Dict:
        """Test if higher capital ratios lead to smaller ECL increases"""
        try:
            period_info = self.metrics.stress_periods[period]
            
            # Filter for stress period
            stress_df = df[
                (df['date'] >= pd.to_datetime(period_info['start'])) &
                (df['date'] <= pd.to_datetime(period_info['end']))
            ].copy()
            
            # Remove invalid values
            stress_df = stress_df.replace([np.inf, -np.inf], np.nan).dropna(
                subset=['tier1_ratio', f'ecl_coverage_change_{period}']
            )
            
            # Calculate median Tier 1 ratio
            median_tier1 = stress_df['tier1_ratio'].median()
            
            # Split banks by Tier 1 ratio
            high_capital = stress_df[stress_df['tier1_ratio'] > median_tier1]
            low_capital = stress_df[stress_df['tier1_ratio'] <= median_tier1]
            
            # Calculate ECL changes
            metric = f'ecl_coverage_change_{period}'
            
            high_changes = high_capital[metric].dropna()
            low_changes = low_capital[metric].dropna()
            
            # Calculate difference percentage
            difference_pct = ((high_changes.mean() - low_changes.mean()) / 
                            abs(low_changes.mean())) * -100  # Negative because we expect smaller increases
            
            # Perform statistical test
            stat, p_value = stats.mannwhitneyu(
                high_changes,
                low_changes,
                alternative='less'
            )
            
            return {
                'difference_pct': difference_pct,
                'hypothesis_supported': difference_pct >= 15 and p_value < self.significance_level,
                'p_value': p_value,
                'statistic': stat,
                'high_capital_mean': high_changes.mean(),
                'low_capital_mean': low_changes.mean(),
                'median_tier1': median_tier1,
                'sample_size': {
                    'high_capital': len(high_changes),
                    'low_capital': len(low_changes)
                }
            }
            
        except Exception as e:
            print(f"Error testing hypothesis 3: {str(e)}")
            return {}

    def test_hypothesis_4(self, df: pd.DataFrame, period: str) -> Dict:
        """Test if model estimates are within 10% of actual ECL"""
        try:
            period_info = self.metrics.stress_periods[period]
            
            # Filter for stress period
            stress_df = df[
                (df['date'] >= pd.to_datetime(period_info['start'])) &
                (df['date'] <= pd.to_datetime(period_info['end']))
            ].copy()
            
            # Remove invalid values
            stress_df = stress_df.replace([np.inf, -np.inf], np.nan).dropna(
                subset=['lnatres', 'predicted_ecl', 'prediction_error']
            )
            
            # Calculate actual vs. predicted ECL
            actual_ecl = stress_df['lnatres']
            predicted_ecl = stress_df['predicted_ecl']
            
            # Calculate percentage difference
            pct_diff = stress_df['prediction_error'].abs()
            
            # Calculate accuracy metrics
            within_10_pct = (pct_diff <= 10).mean() * 100
            mean_abs_error = pct_diff.mean()
            
            return {
                'within_10_pct': within_10_pct,
                'mean_abs_error': mean_abs_error,
                'hypothesis_supported': within_10_pct >= 90,  # At least 90% of predictions within 10%
                'sample_size': len(actual_ecl),
                'error_distribution': {
                    'mean': pct_diff.mean(),
                    'median': pct_diff.median(),
                    'std': pct_diff.std(),
                    'max': pct_diff.max(),
                    'min': pct_diff.min()
                }
            }
            
        except Exception as e:
            print(f"Error testing hypothesis 4: {str(e)}")
            return {}

class ECLVisualizer:
    def __init__(self):
        self.metrics = ECLMetrics()
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'warning': '#d62728',
            'success': '#2ca02c',
            'background': '#ffffff',
            'grid': '#e6e6e6'
        }
        self.layout_defaults = dict(
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font=dict(color='#333333'),
            height=700,
            margin=dict(l=50, r=20, t=50, b=50)
        )

    def create_hypothesis1_visualization(self, df: pd.DataFrame, period: str) -> go.Figure:
        """Create visualization for CRE concentration hypothesis"""
        period_info = self.metrics.stress_periods[period]
        
        # Filter and prepare data
        stress_df = df[
            (df['date'] >= pd.to_datetime(period_info['start'])) &
            (df['date'] <= pd.to_datetime(period_info['end']))
        ].copy()
        
        # Remove invalid values
        stress_df = stress_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=['cre_concentration', f'ecl_coverage_change_{period}']
        )
        
        # Create quartiles based on CRE concentration
        stress_df['cre_quartile'] = pd.qcut(
            stress_df['cre_concentration'], 
            q=4, 
            labels=['Q1', 'Q2', 'Q3', 'Q4']
        )
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "ECL Changes by CRE Concentration Quartile",
                "Time Series of ECL Changes",
                "CRE Concentration Distribution",
                "ECL vs CRE Concentration Scatter"
            ),
            vertical_spacing=0.15
        )
        
        # Box plot of ECL changes by quartile
        fig.add_trace(
            go.Box(
                x=stress_df['cre_quartile'],
                y=stress_df[f'ecl_coverage_change_{period}'],
                name='ECL Changes',
                marker_color=self.color_scheme['primary']
            ),
            row=1, col=1
        )
        
        # Time series by quartile
        for quartile in ['Q1', 'Q4']:
            quartile_data = stress_df[stress_df['cre_quartile'] == quartile]
            quartile_data = quartile_data.sort_values('date')
            fig.add_trace(
                go.Scatter(
                    x=quartile_data['date'],
                    y=quartile_data[f'ecl_coverage_change_{period}'],
                    name=f'Quartile {quartile}',
                    mode='lines+markers'
                ),
                row=1, col=2
            )
        
        # Distribution histogram
        fig.add_trace(
            go.Histogram(
                x=stress_df['cre_concentration'],
                nbinsx=30,
                name='Distribution',
                marker_color=self.color_scheme['secondary']
            ),
            row=2, col=1
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=stress_df['cre_concentration'],
                y=stress_df[f'ecl_coverage_change_{period}'],
                mode='markers',
                marker=dict(
                    color=stress_df['tier1_ratio'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Tier 1 Ratio")
                ),
                name='Banks'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"CRE Concentration Analysis During {period_info['name']}",
            showlegend=True,
            **self.layout_defaults
        )
        
        # Update axes
        fig.update_xaxes(title_text="CRE Concentration (%)", row=2, col=2)
        fig.update_yaxes(title_text="ECL Change (%)", row=2, col=2)
        
        return fig

    def create_hypothesis2_visualization(self, df: pd.DataFrame, period: str) -> go.Figure:
        """Create visualization for consumer loan sensitivity hypothesis"""
        period_info = self.metrics.stress_periods[period]
        
        # Filter and prepare data
        stress_df = df[
            (df['date'] >= pd.to_datetime(period_info['start'])) &
            (df['date'] <= pd.to_datetime(period_info['end']))
        ].copy()
        
        # Remove invalid values
        stress_df = stress_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=['consumer_loan_ratio', f'ecl_coverage_change_{period}']
        )
        
        # Calculate median consumer loan ratio
        median_consumer_ratio = stress_df['consumer_loan_ratio'].median()
        
        # Create high/low consumer loan groups
        stress_df['consumer_group'] = np.where(
            stress_df['consumer_loan_ratio'] > median_consumer_ratio,
            'High Consumer',
            'Low Consumer'
        )
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "ECL Changes by Consumer Loan Concentration",
                "Time Series of ECL Changes",
                "Consumer Loan Ratio Distribution",
                "ECL Change vs Consumer Loan Ratio"
            ),
            vertical_spacing=0.15
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                x=stress_df['consumer_group'],
                y=stress_df[f'ecl_coverage_change_{period}'],
                name='ECL Changes',
                marker_color=self.color_scheme['primary']
            ),
            row=1, col=1
        )
        
        # Time series by group
        for group in ['High Consumer', 'Low Consumer']:
            group_data = stress_df[stress_df['consumer_group'] == group]
            group_data = group_data.sort_values('date')
            fig.add_trace(
                go.Scatter(
                    x=group_data['date'],
                    y=group_data[f'ecl_coverage_change_{period}'],
                    name=group,
                    mode='lines+markers'
                ),
                row=1, col=2
            )
        
        # Distribution histogram
        fig.add_trace(
            go.Histogram(
                x=stress_df['consumer_loan_ratio'],
                nbinsx=30,
                name='Distribution',
                marker_color=self.color_scheme['secondary']
            ),
            row=2, col=1
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=stress_df['consumer_loan_ratio'],
                y=stress_df[f'ecl_coverage_change_{period}'],
                mode='markers',
                marker=dict(
                    color=stress_df['tier1_ratio'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Tier 1 Ratio")
                ),
                name='Banks'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Consumer Loan Analysis During {period_info['name']}",
            showlegend=True,
            **self.layout_defaults
        )
        
        return fig

    def create_hypothesis3_visualization(self, df: pd.DataFrame, period: str) -> go.Figure:
        """Create visualization for capital structure hypothesis"""
        period_info = self.metrics.stress_periods[period]
        
        # Filter and prepare data
        stress_df = df[
            (df['date'] >= pd.to_datetime(period_info['start'])) &
            (df['date'] <= pd.to_datetime(period_info['end']))
        ].copy()
        
        # Remove invalid values
        stress_df = stress_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=['tier1_ratio', f'ecl_coverage_change_{period}']
        )
        
        median_tier1 = stress_df['tier1_ratio'].median()
        stress_df['capital_group'] = np.where(
            stress_df['tier1_ratio'] > median_tier1,
            'High Capital',
            'Low Capital'
        )
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "ECL Changes by Capital Level",
                "Time Series of ECL Changes",
                "Tier 1 Ratio Distribution",
                "ECL vs Tier 1 Ratio"
            ),
            vertical_spacing=0.15
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                x=stress_df['capital_group'],
                y=stress_df[f'ecl_coverage_change_{period}'],
                name='ECL Changes',
                marker_color=self.color_scheme['primary']
            ),
            row=1, col=1
        )
        
        # Time series
        for group in ['High Capital', 'Low Capital']:
            group_data = stress_df[stress_df['capital_group'] == group]
            group_data = group_data.sort_values('date')
            fig.add_trace(
                go.Scatter(
                    x=group_data['date'],
                    y=group_data[f'ecl_coverage_change_{period}'],
                    name=group,
                    mode='lines+markers'
                ),
                row=1, col=2
            )
        
        # Distribution histogram
        fig.add_trace(
            go.Histogram(
                x=stress_df['tier1_ratio'],
                nbinsx=30,
                name='Distribution',
                marker_color=self.color_scheme['secondary']
            ),
            row=2, col=1
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=stress_df['tier1_ratio'],
                y=stress_df[f'ecl_coverage_change_{period}'],
                mode='markers',
                marker=dict(
                    color=stress_df['consumer_loan_ratio'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Consumer Loan Ratio")
                ),
                name='Banks'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Capital Structure Analysis During {period_info['name']}",
            showlegend=True,
            **self.layout_defaults
        )
        
        return fig

    def create_hypothesis4_visualization(self, df: pd.DataFrame, period: str) -> go.Figure:
        """Create visualization for ECL model accuracy hypothesis"""
        period_info = self.metrics.stress_periods[period]
        
        # Filter and prepare data
        stress_df = df[
            (df['date'] >= pd.to_datetime(period_info['start'])) &
            (df['date'] <= pd.to_datetime(period_info['end']))
        ].copy()
        
        # Remove invalid values
        stress_df = stress_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=['lnatres', 'predicted_ecl', 'prediction_error', 'asset']
        )
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Predicted vs Actual ECL",
                "Prediction Error Distribution",
                "Error by Bank Size",
                "Time Series of Prediction Error"
            ),
            vertical_spacing=0.15
        )
        
        # Scatter plot of predicted vs actual
        fig.add_trace(
            go.Scatter(
                x=stress_df['lnatres'],
                y=stress_df['predicted_ecl'],
                mode='markers',
                name='Predictions'
            ),
            row=1, col=1
        )
        
        # Add 45-degree line
        max_val = max(stress_df['lnatres'].max(), stress_df['predicted_ecl'].max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash')
            ),
            row=1, col=1
        )
        
        # Error distribution
        fig.add_trace(
            go.Histogram(
                x=stress_df['prediction_error'],
                nbinsx=30,
                name='Error Distribution'
            ),
            row=1, col=2
        )
        
        # Error by bank size
        fig.add_trace(
            go.Scatter(
                x=stress_df['asset'],
                y=stress_df['prediction_error'],
                mode='markers',
                name='Error by Size'
            ),
            row=2, col=1
        )
        
        # Time series of error
        fig.add_trace(
            go.Scatter(
                x=stress_df['date'],
                y=stress_df['prediction_error'],
                mode='lines+markers',
                name='Error Over Time'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"ECL Model Accuracy Analysis During {period_info['name']}",
            showlegend=True,
            **self.layout_defaults
        )
        
        return fig

class ECLDashboard:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.visualizer = ECLVisualizer()
        self.hypothesis_tester = HypothesisTester()
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("ECL Research Analysis Dashboard",
                           className="text-primary mb-4"),
                    html.H5("Testing Expected Credit Loss Determinants",
                           className="text-muted mb-4")
                ])
            ]),

            # Control Panel and Main Content
            dbc.Row([
                # Control Panel
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Analysis Controls"),
                        dbc.CardBody([
                            html.Label("Select Stress Period"),
                            dcc.Dropdown(
                                id='stress-period-selector',
                                options=[
                                    {'label': 'Financial Crisis (2008-2010)', 
                                     'value': 'gfc'},
                                    {'label': 'COVID-19 (2020-2021)', 
                                     'value': 'covid'}
                                ],
                                value='gfc',
                                className="mb-3"
                            ),
                            
                            html.Label("Select Hypothesis"),
                            dcc.Dropdown(
                                id='hypothesis-selector',
                                options=[
                                    {'label': 'H1: CRE Concentration Impact', 
                                     'value': 'h1'},
                                    {'label': 'H2: Consumer Loan Sensitivity', 
                                     'value': 'h2'},
                                    {'label': 'H3: Capital Structure Effect', 
                                     'value': 'h3'},
                                    {'label': 'H4: ECL Model Accuracy', 
                                     'value': 'h4'}
                                ],
                                value='h1',
                                className="mb-3"
                            ),
                            
                            html.Div(id='hypothesis-description', 
                                    className="mt-3 small")
                        ])
                    ], className="mb-4"),
                    
                    # Bank Filter Panel
                    dbc.Card([
                        dbc.CardHeader("Bank Filters"),
                        dbc.CardBody([
                            html.Label("Asset Size Range"),
                            dcc.RangeSlider(
                                id='asset-size-filter',
                                min=0,
                                max=3000,
                                step=100,
                                value=[0, 3000],
                                marks={i: f'${i}B' for i in range(0, 3001, 500)}
                            ),
                            
                            html.Label("Bank Type", className="mt-3"),
                            dcc.Checklist(
                                id='bank-type-filter',
                                options=[
                                    {'label': ' Commercial Banks', 'value': 'commercial'},
                                    {'label': ' Consumer Banks', 'value': 'consumer'},
                                    {'label': ' Investment Banks', 'value': 'investment'}
                                ],
                                value=['commercial', 'consumer', 'investment']
                            )
                        ])
                    ], className="mb-4"),

                    # Bank List Panel
                    dbc.Card([
                        dbc.CardHeader("Banks Used in Analysis"),
                        dbc.CardBody([
                            dbc.ListGroup(
                                id='bank-list',
                                flush=True
                            )
                        ])
                    ], className="mb-4"),

                ], width=3),

                # Main Visualization Area
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Row([
                                dbc.Col(html.H5("Analysis Results", className="mb-0"), width=8),
                                dbc.Col(
                                    dbc.Button(
                                        "Download Data",
                                        id="download-button",
                                        color="primary",
                                        size="sm",
                                        className="float-right"
                                    ),
                                    width=4,
                                    className="text-right"
                                )
                            ])
                        ]),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-1",
                                type="default",
                                children=[dcc.Graph(id='main-visualization')]
                            )
                        ])
                    ])
                ], width=9)
            ]),

            # Results Panels
            dbc.Row([
                # Statistical Results
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Statistical Analysis"),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-2",
                                type="default",
                                children=[html.Div(id='statistical-results')]
                            )
                        ])
                    ])
                ], width=6),

                # Supporting Metrics
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Supporting Analysis"),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-3",
                                type="default",
                                children=[html.Div(id='supporting-metrics')]
                            )
                        ])
                    ])
                ], width=6)
            ], className="mt-4"),
            
            # Download Component
            dcc.Download(id="download-data")
            
        ], fluid=True)

    def setup_callbacks(self):
        @self.app.callback(
            [Output('main-visualization', 'figure'),
             Output('statistical-results', 'children'),
             Output('supporting-metrics', 'children'),
             Output('hypothesis-description', 'children'),
             Output('bank-list', 'children')],
            [Input('stress-period-selector', 'value'),
             Input('hypothesis-selector', 'value'),
             Input('asset-size-filter', 'value'),
             Input('bank-type-filter', 'value')]
        )
        def update_analysis(stress_period: str, hypothesis: str, 
                          asset_range: List[float], bank_types: List[str]):
            """Update all visualizations and analysis based on user selections"""
            
            # Filter data based on selections
            filtered_df = self.filter_data(asset_range, bank_types)
            
            # Generate visualizations and results based on hypothesis
            if hypothesis == 'h1':
                main_fig = self.visualizer.create_hypothesis1_visualization(
                    filtered_df, stress_period
                )
                results = self.hypothesis_tester.test_hypothesis_1(
                    filtered_df, stress_period
                )
                description = self.get_hypothesis_description('h1')
                
            elif hypothesis == 'h2':
                main_fig = self.visualizer.create_hypothesis2_visualization(
                    filtered_df, stress_period
                )
                results = self.hypothesis_tester.test_hypothesis_2(
                    filtered_df, stress_period
                )
                description = self.get_hypothesis_description('h2')
                
            elif hypothesis == 'h3':
                main_fig = self.visualizer.create_hypothesis3_visualization(
                    filtered_df, stress_period
                )
                results = self.hypothesis_tester.test_hypothesis_3(
                    filtered_df, stress_period
                )
                description = self.get_hypothesis_description('h3')
                
            elif hypothesis == 'h4':
                main_fig = self.visualizer.create_hypothesis4_visualization(
                    filtered_df, stress_period
                )
                results = self.hypothesis_tester.test_hypothesis_4(
                    filtered_df, stress_period
                )
                description = self.get_hypothesis_description('h4')
                
            else:
                main_fig = go.Figure()
                results = {}
                description = ""
            
            stats_panel = self.create_stats_panel(results)
            supporting_metrics = self.create_supporting_metrics_panel(
                filtered_df, stress_period, hypothesis
            )

            # Generate bank list
            bank_list_items = [
                dbc.ListGroupItem(bank) 
                for bank in sorted(filtered_df['abbreviated_name'].unique())
            ]
            bank_list = bank_list_items if bank_list_items else [dbc.ListGroupItem("No banks available")]

            return main_fig, stats_panel, supporting_metrics, description, bank_list

        @self.app.callback(
            Output("download-data", "data"),
            Input("download-button", "n_clicks"),
            [State('stress-period-selector', 'value'),
             State('hypothesis-selector', 'value'),
             State('asset-size-filter', 'value'),
             State('bank-type-filter', 'value')]
        )
        def download_results(n_clicks, stress_period, hypothesis, 
                           asset_range, bank_types):
            """Generate downloadable results based on current selections"""
            if n_clicks is None:
                return None
                
            filtered_df = self.filter_data(asset_range, bank_types)
            
            # Prepare data for download
            download_df = self.prepare_download_data(
                filtered_df, stress_period, hypothesis
            )
            
            return dcc.send_data_frame(
                download_df.to_excel,
                "ecl_analysis_results.xlsx",
                sheet_name="Results"
            )

    def filter_data(self, asset_range: List[float], 
                   bank_types: List[str]) -> pd.DataFrame:
        """Filter DataFrame based on user selections"""
        df = self.df.copy()
        
        # Apply asset size filter
        asset_min = asset_range[0] * 1e9  # Convert billions to actual values
        asset_max = asset_range[1] * 1e9
        df = df[(df['asset'] >= asset_min) & (df['asset'] <= asset_max)]
        
        # Apply bank type filter
        if bank_types:
            type_masks = []
            if 'commercial' in bank_types:
                type_masks.append(df['consumer_loan_ratio'] <= 50)
            if 'consumer' in bank_types:
                type_masks.append(df['consumer_loan_ratio'] > 50)
            if 'investment' in bank_types:
                type_masks.append(df['lnci'] / df['lnlsgr'] > 0.5)
            
            combined_mask = type_masks[0]
            for mask in type_masks[1:]:
                combined_mask = combined_mask | mask
            
            df = df[combined_mask]
        
        return df

    def get_hypothesis_description(self, hypothesis: str) -> str:
        """Return description of selected hypothesis"""
        descriptions = {
            'h1': "Banks with high CRE concentration (top quartile) experience at least 20% larger ECL increases during stress periods compared to those in the bottom quartile.",
            'h2': "Banks with above-median consumer loan ratios show greater ECL sensitivity during stress periods.",
            'h3': "Banks with above-median Tier 1 capital ratios experience at least 15% smaller ECL increases during stress periods.",
            'h4': "A standardized model can estimate ECL within 10% of actual FDIC-reported figures."
        }
        return descriptions.get(hypothesis, "")

    def create_supporting_metrics_panel(self, df: pd.DataFrame, 
                                     period: str, hypothesis: str) -> html.Div:
        """Create supporting metrics visualization based on hypothesis"""
        metrics = {}
        
        if hypothesis == 'h1':
            metrics = {
                'Total Banks': len(df['cert'].unique()),
                'Median CRE Concentration': f"{df['cre_concentration'].median():.1f}%",
                'Mean ECL Change': f"{df[f'ecl_coverage_change_{period}'].mean():.1f}%",
                'Correlation': f"{df['cre_concentration'].corr(df[f'ecl_coverage_change_{period}']):.3f}"
            }
        elif hypothesis == 'h2':
            median_consumer_ratio = df['consumer_loan_ratio'].median()
            metrics = {
                'High Consumer Banks': len(df[df['consumer_loan_ratio'] > median_consumer_ratio]['cert'].unique()),
                'Low Consumer Banks': len(df[df['consumer_loan_ratio'] <= median_consumer_ratio]['cert'].unique()),
                'Median Consumer Ratio': f"{median_consumer_ratio:.1f}%"
            }
        elif hypothesis == 'h3':
            metrics = {
                'Median Tier 1 Ratio': f"{df['tier1_ratio'].median():.1f}%",
                'Mean ECL Change': f"{df[f'ecl_coverage_change_{period}'].mean():.1f}%",
                'Capital-ECL Correlation': f"{df['tier1_ratio'].corr(df[f'ecl_coverage_change_{period}']):.3f}"
            }
        elif hypothesis == 'h4':
            # Remove invalid values
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['prediction_error'])
            metrics = {
                'Mean Prediction Error': f"{df['prediction_error'].mean():.1f}%",
                'Median Prediction Error': f"{df['prediction_error'].median():.1f}%",
                'Error Std Dev': f"{df['prediction_error'].std():.1f}%"
            }
            
        return html.Div([
            html.H6("Supporting Metrics"),
            html.Hr(),
            *[html.P([html.Strong(f"{k}: "), v]) for k, v in metrics.items()]
        ])

    def create_stats_panel(self, results: Dict) -> html.Div:
        """Create statistical results panel"""
        if not results:
            return html.Div("No results available.")
        
        content = [
            html.H6("Statistical Results"),
            html.Hr()
        ]
        for key, value in results.items():
            if isinstance(value, dict):
                content.append(html.P([html.Strong(f"{key}:")]))
                for sub_key, sub_value in value.items():
                    content.append(html.P(f" - {sub_key}: {sub_value}"))
            else:
                content.append(html.P([html.Strong(f"{key}: "), str(value)]))
        
        return html.Div(content)

    def prepare_download_data(self, df: pd.DataFrame, stress_period: str, hypothesis: str) -> pd.DataFrame:
        """Prepare DataFrame for download based on current analysis"""
        # Select relevant columns based on hypothesis
        columns = ['cert', 'date']
        if hypothesis == 'h1':
            columns.extend(['cre_concentration', f'ecl_coverage_change_{stress_period}'])
        elif hypothesis == 'h2':
            columns.extend(['consumer_loan_ratio', f'ecl_coverage_change_{stress_period}'])
        elif hypothesis == 'h3':
            columns.extend(['tier1_ratio', f'ecl_coverage_change_{stress_period}'])
        elif hypothesis == 'h4':
            columns.extend(['lnatres', 'predicted_ecl', 'prediction_error'])
        
        return df[columns]

    def main(self):
        self.app.run_server(debug=False)

def main():
    # Initialize data extraction and processing
    processor = ECLDataProcessor()
    
    # Define specific banks to analyze
    bank_list = [
        {"cert": "3511", "name": "Wells Fargo Bank, National Association", "abbreviated_name": "Wells Fargo"},
        {"cert": "3510", "name": "Bank of America, National Association", "abbreviated_name": "Bank of America"},
        {"cert": "7213", "name": "Citibank, National Association", "abbreviated_name": "Citibank"},
        {"cert": "628", "name": "JPMorgan Chase Bank, National Association", "abbreviated_name": "JPMorgan Chase"},
        {"cert": "6548", "name": "U.S. Bank National Association", "abbreviated_name": "U.S. Bank"},
        {"cert": "6384", "name": "PNC Bank, National Association", "abbreviated_name": "PNC Bank"},
        {"cert": "9846", "name": "Truist Bank", "abbreviated_name": "Truist"},
        {"cert": "33124", "name": "Goldman Sachs Bank USA", "abbreviated_name": "Goldman Sachs"},
        {"cert": "32992", "name": "Morgan Stanley Bank, National Association", "abbreviated_name": "Morgan Stanley"},
        {"cert": "18409", "name": "TD Bank, National Association", "abbreviated_name": "TD Bank"},
        {"cert": "4297", "name": "Capital One, National Association", "abbreviated_name": "Capital One"},
        {"cert": "639", "name": "The Bank of New York Mellon", "abbreviated_name": "BNY Mellon"},
        {"cert": "6672", "name": "Fifth Third Bank, National Association", "abbreviated_name": "Fifth Third Bank"},
        {"cert": "57957", "name": "Citizens Bank, National Association", "abbreviated_name": "Citizens Bank"},
        {"cert": "57803", "name": "Ally Bank", "abbreviated_name": "Ally Bank"},
        {"cert": "17534", "name": "KeyBank National Association", "abbreviated_name": "KeyBank"},
        {"cert": "5649", "name": "Discover Bank", "abbreviated_name": "Discover Bank"},
        {"cert": "27314", "name": "Synchrony Bank", "abbreviated_name": "Synchrony Bank"},
        {"cert": "29950", "name": "Santander Bank, N.A.", "abbreviated_name": "Santander Bank"}
    ]
    
    start_date = '20080101'
    end_date = '20240630'
    
    # Process data for each bank
    all_data = []
    for bank in bank_list:
        params = {
            "filters": f"CERT:{bank['cert']} AND REPDTE:[{start_date} TO {end_date}]",
            "fields": (
                "CERT,REPDTE,ASSET,DEP,LNLSGR,LNLSNET,SC,LNRE,LNCI,LNAG,LNCRCD,LNCONOTH,"
                "LNATRES,P3ASSET,P9ASSET,RBCT1J,DRLNLS,CRLNLS,NETINC,ERNASTR,NPERFV,"
                "P3ASSETR,P9ASSETR,NIMY,NTLNLSR,LNATRESR,NCLNLSR,ROA,ROE,RBC1AAJ,"
                "RBCT2,RBCRWAJ,LNLSDEPR,LNLSNTV,EEFFR,LNRESNCR,ELNANTR,IDERNCVR,NTLNLSQ,"
                "LNRECONS,LNRENRES,LNRENROW,LNRENROT,LNRERES,LNREMULT,LNREAG,LNRECNFM,"
                "LNRECNOT,LNCOMRE,CT1BADJ,EQ,EQPP"
            ),
            "limit": 10000
        }
        
        try:
            response = requests.get(
                f"{BASE_URL}/financials", 
                params=params, 
                headers={"Accept": "application/json"}, 
                verify=False
            )
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data:
                bank_data = [
                    {**item['data'], 'bank': bank['name'], 'abbreviated_name': bank['abbreviated_name']} 
                    for item in data['data'] 
                    if isinstance(item, dict) and 'data' in item
                ]
                all_data.extend(bank_data)
        except Exception as e:
            continue
    
    if not all_data:
        return
    
    df = processor.process_raw_data(all_data)
    
    if df.empty:
        return
    
    df = processor.calculate_ecl_metrics(df)
    df = processor.calculate_stress_period_changes(df)
    df = processor.calculate_predicted_ecl_and_error(df)
    
    dashboard = ECLDashboard(df)
    dashboard.main()

if __name__ == "__main__":
    warnings.filterwarnings('ignore', message='.*Unverified HTTPS.*')
    main()
