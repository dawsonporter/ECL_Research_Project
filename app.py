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
                'LNREAG',   # Agricultural Real Estate Loans
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

        # Define bank groups and mappings
        self.national_banks = [
            "Wells Fargo Bank, National Association",
            "Bank of America, National Association",
            "Citibank, National Association",
            "JPMorgan Chase Bank, National Association",
            "U.S. Bank National Association",
            "PNC Bank, National Association",
            "Truist Bank",
            "Goldman Sachs Bank USA",
            "Morgan Stanley Bank, National Association",
            "TD Bank, National Association",
            "Capital One, National Association",
            "Fifth Third Bank, National Association",
            "Citizens Bank, National Association",
            "Ally Bank",
            "KeyBank National Association"
        ]

        self.regional_banks = [
            "Associated Bank, National Association",
            "BOKF, National Association",
            "BankUnited, National Association",
            "City National Bank of Florida",
            "EverBank, National Association",
            "First National Bank of Pennsylvania",
            "Old National Bank",
            "SoFi Bank, National Association",
            "Trustmark National Bank",
            "Webster Bank, National Association",
            "Wintrust Bank, National Association",
            "Zions Bancorporation, N.A.",
            "Fulton Bank, National Association",
            "SouthState Bank, National Association",
            "UMB Bank, National Association",
            "Valley National Bank",
            "Bremer Bank, National Association",
            "The Bank of New York Mellon"
        ]

        self.nebraska_banks = [
            "Dundee Bank",
            "AMERICAN NATIONAL BANK",
            "FIVE POINTS BANK",
            "SECURITY FIRST BANK",
            "SECURITY NATIONAL BANK OF OMAHA",
            "FRONTIER BANK",
            "WEST GATE BANK",
            "CORE BANK",
            "FIRST STATE BANK NEBRASKA",
            "ACCESS BANK",
            "CORNHUSKER BANK",
            "ARBOR BANK",
            "WASHINGTON COUNTY BANK",
            "ENTERPRISE BANK",
            "PREMIER BANK NATIONAL ASSOCIATION",
            "FIRST WESTROADS BANK, INC."
        ]

        self.bank_name_mapping = {
            "First National Bank of Omaha": "FNBO",
            "Associated Bank, National Association": "Associated Bank",
            "BOKF, National Association": "BOKF",
            "BankUnited, National Association": "BankUnited",
            "City National Bank of Florida": "City National Bank of Florida",
            "EverBank, National Association": "EverBank",
            "First National Bank of Pennsylvania": "First National Bank of PA",
            "Old National Bank": "Old National Bank",
            "SoFi Bank, National Association": "SoFi Bank",
            "Trustmark National Bank": "Trustmark Bank",
            "Webster Bank, National Association": "Webster Bank",
            "Wintrust Bank, National Association": "Wintrust Bank",
            "Zions Bancorporation, N.A.": "Zions Bank",
            "Capital One, National Association": "Capital One",
            "Discover Bank": "Discover Bank",
            "Comenity Bank": "Comenity Bank",
            "Synchrony Bank": "Synchrony Bank",
            "Wells Fargo Bank, National Association": "Wells Fargo",
            "U.S. Bank National Association": "U.S. Bank",
            "Fulton Bank, National Association": "Fulton Bank",
            "SouthState Bank, National Association": "SouthState Bank",
            "UMB Bank, National Association": "UMB Bank",
            "Valley National Bank": "Valley National Bank",
            "Bremer Bank, National Association": "Bremer Bank",
            "The Bank of New York Mellon": "BNY Mellon",
            "Commerce Bank": "Commerce Bank",
            "Frost Bank": "Frost Bank",
            "FirstBank": "FirstBank",
            "Pinnacle Bank": "Pinnacle Bank",
            "Dundee Bank": "Dundee Bank",
            "American National Bank": "American National",
            "Five Points Bank": "Five Points Bank",
            "Security First Bank": "Security First Bank",
            "Security National Bank of Omaha": "Security National",
            "Frontier Bank": "Frontier Bank",
            "West Gate Bank": "West Gate Bank",
            "Core Bank": "Core Bank",
            "First State Bank Nebraska": "First State Bank",
            "Access Bank": "Access Bank",
            "Cornhusker Bank": "Cornhusker Bank",
            "Arbor Bank": "Arbor Bank",
            "Washington County Bank": "Washington County",
            "Enterprise Bank": "Enterprise Bank",
            "Premier Bank National Association": "Premier Bank",
            "First Westroads Bank, Inc.": "First Westroads"
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
        
        # Add bank type classification
        df['bank_type'] = df['bank'].apply(self.classify_bank_type)
        
        return df

    def classify_bank_type(self, bank_name: str) -> str:
        """Classify banks into their respective groups"""
        if bank_name in self.metrics.national_banks:
            return 'National Systemic Bank'
        elif bank_name in self.metrics.regional_banks:
            return 'Regional Bank'
        elif bank_name in self.metrics.nebraska_banks:
            return 'Nebraska Bank'
        else:
            return 'Other'

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
            'grid': '#e6e6e6',
            'national': '#003f5c',
            'regional': '#58508d',
            'nebraska': '#bc5090'
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
                "ECL Changes by CRE Concentration Quartile and Bank Type",
                "Time Series of ECL Changes",
                "CRE Concentration Distribution by Bank Type",
                "ECL vs CRE Concentration by Bank Type"
            ),
            vertical_spacing=0.15
        )
        
        # Box plot of ECL changes by quartile and bank type
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            fig.add_trace(
                go.Box(
                    x=bank_data['cre_quartile'],
                    y=bank_data[f'ecl_coverage_change_{period}'],
                    name=bank_type,
                    marker_color=self.color_scheme[bank_type.split()[0].lower()]
                ),
                row=1, col=1
            )
        
        # Time series by bank type
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            bank_data = bank_data.sort_values('date')
            fig.add_trace(
                go.Scatter(
                    x=bank_data['date'],
                    y=bank_data[f'ecl_coverage_change_{period}'],
                    name=bank_type,
                    mode='lines+markers',
                    line=dict(color=self.color_scheme[bank_type.split()[0].lower()])
                ),
                row=1, col=2
            )
        
        # Distribution histogram by bank type
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            fig.add_trace(
                go.Histogram(
                    x=bank_data['cre_concentration'],
                    name=bank_type,
                    marker_color=self.color_scheme[bank_type.split()[0].lower()],
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Scatter plot by bank type
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            fig.add_trace(
                go.Scatter(
                    x=bank_data['cre_concentration'],
                    y=bank_data[f'ecl_coverage_change_{period}'],
                    mode='markers',
                    name=bank_type,
                    marker=dict(
                        color=self.color_scheme[bank_type.split()[0].lower()],
                        size=10,
                        opacity=0.7
                    )
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text=f"CRE Concentration Analysis During {period_info['name']} by Bank Type",
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
                "ECL Changes by Consumer Loan Concentration and Bank Type",
                "Time Series of ECL Changes by Bank Type",
                "Consumer Loan Ratio Distribution by Bank Type",
                "ECL Change vs Consumer Loan Ratio by Bank Type"
            ),
            vertical_spacing=0.15
        )
        
        # Box plot by bank type
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            fig.add_trace(
                go.Box(
                    x=bank_data['consumer_group'],
                    y=bank_data[f'ecl_coverage_change_{period}'],
                    name=bank_type,
                    marker_color=self.color_scheme[bank_type.split()[0].lower()]
                ),
                row=1, col=1
            )
        
        # Time series by bank type
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            bank_data = bank_data.sort_values('date')
            fig.add_trace(
                go.Scatter(
                    x=bank_data['date'],
                    y=bank_data[f'ecl_coverage_change_{period}'],
                    name=bank_type,
                    mode='lines+markers',
                    line=dict(color=self.color_scheme[bank_type.split()[0].lower()])
                ),
                row=1, col=2
            )
        
        # Distribution histogram by bank type
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            fig.add_trace(
                go.Histogram(
                    x=bank_data['consumer_loan_ratio'],
                    name=bank_type,
                    marker_color=self.color_scheme[bank_type.split()[0].lower()],
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Scatter plot by bank type
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            fig.add_trace(
                go.Scatter(
                    x=bank_data['consumer_loan_ratio'],
                    y=bank_data[f'ecl_coverage_change_{period}'],
                    mode='markers',
                    name=bank_type,
                    marker=dict(
                        color=self.color_scheme[bank_type.split()[0].lower()],
                        size=10,
                        opacity=0.7
                    )
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text=f"Consumer Loan Analysis During {period_info['name']} by Bank Type",
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
                "ECL Changes by Capital Level and Bank Type",
                "Time Series of ECL Changes by Bank Type",
                "Tier 1 Ratio Distribution by Bank Type",
                "ECL vs Tier 1 Ratio by Bank Type"
            ),
            vertical_spacing=0.15
        )
        
        # Box plot by bank type
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            fig.add_trace(
                go.Box(
                    x=bank_data['capital_group'],
                    y=bank_data[f'ecl_coverage_change_{period}'],
                    name=bank_type,
                    marker_color=self.color_scheme[bank_type.split()[0].lower()]
                ),
                row=1, col=1
            )
        
        # Time series by bank type
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            bank_data = bank_data.sort_values('date')
            fig.add_trace(
                go.Scatter(
                    x=bank_data['date'],
                    y=bank_data[f'ecl_coverage_change_{period}'],
                    name=bank_type,
                    mode='lines+markers',
                    line=dict(color=self.color_scheme[bank_type.split()[0].lower()])
                ),
                row=1, col=2
            )
        
        # Distribution histogram by bank type
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            fig.add_trace(
                go.Histogram(
                    x=bank_data['tier1_ratio'],
                    name=bank_type,
                    marker_color=self.color_scheme[bank_type.split()[0].lower()],
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Scatter plot by bank type
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            fig.add_trace(
                go.Scatter(
                    x=bank_data['tier1_ratio'],
                    y=bank_data[f'ecl_coverage_change_{period}'],
                    mode='markers',
                    name=bank_type,
                    marker=dict(
                        color=self.color_scheme[bank_type.split()[0].lower()],
                        size=10,
                        opacity=0.7
                    )
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text=f"Capital Structure Analysis During {period_info['name']} by Bank Type",
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
                "Predicted vs Actual ECL by Bank Type",
                "Prediction Error Distribution by Bank Type",
                "Error by Bank Size and Type",
                "Time Series of Prediction Error by Bank Type"
            ),
            vertical_spacing=0.15
        )
        
        # Scatter plot of predicted vs actual by bank type
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            fig.add_trace(
                go.Scatter(
                    x=bank_data['lnatres'],
                    y=bank_data['predicted_ecl'],
                    mode='markers',
                    name=bank_type,
                    marker=dict(
                        color=self.color_scheme[bank_type.split()[0].lower()],
                        size=10,
                        opacity=0.7
                    )
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
                line=dict(dash='dash', color='black')
            ),
            row=1, col=1
        )
        
        # Error distribution by bank type
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            fig.add_trace(
                go.Histogram(
                    x=bank_data['prediction_error'],
                    name=bank_type,
                    marker_color=self.color_scheme[bank_type.split()[0].lower()],
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        # Error by bank size and type
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            fig.add_trace(
                go.Scatter(
                    x=bank_data['asset'],
                    y=bank_data['prediction_error'],
                    mode='markers',
                    name=bank_type,
                    marker=dict(
                        color=self.color_scheme[bank_type.split()[0].lower()],
                        size=10,
                        opacity=0.7
                    )
                ),
                row=2, col=1
            )
        
        # Time series of error by bank type
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            bank_data = bank_data.sort_values('date')
            fig.add_trace(
                go.Scatter(
                    x=bank_data['date'],
                    y=bank_data['prediction_error'],
                    mode='lines+markers',
                    name=bank_type,
                    line=dict(color=self.color_scheme[bank_type.split()[0].lower()])
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text=f"ECL Model Accuracy Analysis During {period_info['name']} by Bank Type",
            showlegend=True,
            **self.layout_defaults
        )
        
        return fig

class ECLDashboard:
    def __init__(self, df: pd.DataFrame, app: dash.Dash):
        self.df = df
        self.app = app
        self.visualizer = ECLVisualizer()
        self.hypothesis_tester = HypothesisTester()
        self.metrics = ECLMetrics()  # Add this line to initialize metrics
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
                                    {'label': ' National Systemic Banks', 'value': 'national'},
                                    {'label': ' Regional Banks', 'value': 'regional'},
                                    {'label': ' Nebraska Banks', 'value': 'nebraska'}
                                ],
                                value=['national', 'regional', 'nebraska']
                            )
                        ])
                    ], className="mb-4"),

                    # Bank List Panel with Modern Styling
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("Banks by Category", className="mb-0"),
                        ]),
                        dbc.CardBody([
                            # National Banks Section
                            html.Div([
                                html.H6("National Systemic Banks", 
                                       className="text-primary mb-2"),
                                dbc.ListGroup(
                                    id='national-bank-list',
                                    flush=True,
                                    className="mb-3"
                                )
                            ]),
                            # Regional Banks Section
                            html.Div([
                                html.H6("Regional Banks", 
                                       className="text-primary mb-2"),
                                dbc.ListGroup(
                                    id='regional-bank-list',
                                    flush=True,
                                    className="mb-3"
                                )
                            ]),
                            # Nebraska Banks Section
                            html.Div([
                                html.H6("Nebraska Banks", 
                                       className="text-primary mb-2"),
                                dbc.ListGroup(
                                    id='nebraska-bank-list',
                                    flush=True
                                )
                            ])
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
             Output('national-bank-list', 'children'),
             Output('regional-bank-list', 'children'),
             Output('nebraska-bank-list','children')],
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

            # Generate bank lists by type
            national_banks = [
                dbc.ListGroupItem(self.metrics.bank_name_mapping.get(bank, bank), 
                                className="py-2") 
                for bank in sorted(filtered_df[filtered_df['bank_type'] == 'National Systemic Bank']['bank'].unique())
            ]
            national_list = national_banks if national_banks else [dbc.ListGroupItem("No national banks available")]

            regional_banks = [
                dbc.ListGroupItem(self.metrics.bank_name_mapping.get(bank, bank), 
                                className="py-2") 
                for bank in sorted(filtered_df[filtered_df['bank_type'] == 'Regional Bank']['bank'].unique())
            ]
            regional_list = regional_banks if regional_banks else [dbc.ListGroupItem("No regional banks available")]

            nebraska_banks = [
                dbc.ListGroupItem(self.metrics.bank_name_mapping.get(bank, bank), 
                                className="py-2") 
                for bank in sorted(filtered_df[filtered_df['bank_type'] == 'Nebraska Bank']['bank'].unique())
            ]
            nebraska_list = nebraska_banks if nebraska_banks else [dbc.ListGroupItem("No Nebraska banks available")]

            return main_fig, stats_panel, supporting_metrics, description, national_list, regional_list, nebraska_list

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
            type_mapping = {
                'national': 'National Systemic Bank',
                'regional': 'Regional Bank',
                'nebraska': 'Nebraska Bank'
            }
            selected_types = [type_mapping[t] for t in bank_types if t in type_mapping]
            df = df[df['bank_type'].isin(selected_types)]
        
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
        
        # Add bank type breakdown
        bank_type_counts = df['bank_type'].value_counts()
        metrics['Bank Type Distribution'] = {
            'National Systemic Banks': f"{bank_type_counts.get('National Systemic Bank', 0)}",
            'Regional Banks': f"{bank_type_counts.get('Regional Bank', 0)}",
            'Nebraska Banks': f"{bank_type_counts.get('Nebraska Bank', 0)}"
        }
        
        if hypothesis == 'h1':
            metrics['Analysis Metrics'] = {
                'Total Banks': len(df['cert'].unique()),
                'Median CRE Concentration': f"{df['cre_concentration'].median():.1f}%",
                'Mean ECL Change': f"{df[f'ecl_coverage_change_{period}'].mean():.1f}%",
                'Correlation': f"{df['cre_concentration'].corr(df[f'ecl_coverage_change_{period}']):.3f}"
            }
        elif hypothesis == 'h2':
            median_consumer_ratio = df['consumer_loan_ratio'].median()
            metrics['Analysis Metrics'] = {
                'High Consumer Banks': len(df[df['consumer_loan_ratio'] > median_consumer_ratio]['cert'].unique()),
                'Low Consumer Banks': len(df[df['consumer_loan_ratio'] <= median_consumer_ratio]['cert'].unique()),
                'Median Consumer Ratio': f"{median_consumer_ratio:.1f}%"
            }
        elif hypothesis == 'h3':
            metrics['Analysis Metrics'] = {
                'Median Tier 1 Ratio': f"{df['tier1_ratio'].median():.1f}%",
                'Mean ECL Change': f"{df[f'ecl_coverage_change_{period}'].mean():.1f}%",
                'Capital-ECL Correlation': f"{df['tier1_ratio'].corr(df[f'ecl_coverage_change_{period}']):.3f}"
            }
        elif hypothesis == 'h4':
            # Remove invalid values
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['prediction_error'])
            metrics['Analysis Metrics'] = {
                'Mean Prediction Error': f"{df['prediction_error'].mean():.1f}%",
                'Median Prediction Error': f"{df['prediction_error'].median():.1f}%",
                'Error Std Dev': f"{df['prediction_error'].std():.1f}%"
            }
            
        # Create formatted output
        content = []
        for section_name, section_metrics in metrics.items():
            content.extend([
                html.H6(section_name),
                html.Hr(),
                *[html.P([html.Strong(f"{k}: "), v]) for k, v in section_metrics.items()],
                html.Br()
            ])
            
        return html.Div(content)

    def create_stats_panel(self, results: Dict) -> html.Div:
        """Create statistical results panel"""
        if not results:
            return html.Div("No results available.")
        
        content = [
            html.H6("Statistical Results"),
            html.Hr()
        ]
        
        # Format results by bank type if available
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
        columns = ['cert', 'date', 'bank_type', 'bank']
        if hypothesis == 'h1':
            columns.extend(['cre_concentration', f'ecl_coverage_change_{stress_period}'])
        elif hypothesis == 'h2':
            columns.extend(['consumer_loan_ratio', f'ecl_coverage_change_{stress_period}'])
        elif hypothesis == 'h3':
            columns.extend(['tier1_ratio', f'ecl_coverage_change_{stress_period}'])
        elif hypothesis == 'h4':
            columns.extend(['lnatres', 'predicted_ecl', 'prediction_error'])
        
        # Map bank names to abbreviated versions
        download_df = df[columns].copy()
        download_df['bank'] = download_df['bank'].map(
            self.metrics.bank_name_mapping
        ).fillna(download_df['bank'])
        
        return download_df

def get_data():
    """Initialize data extraction and processing"""
    processor = ECLDataProcessor()
    metrics = ECLMetrics()

    # Define bank details including CERT numbers
    bank_details = [
        # National Systemic Banks
        {"cert": "3511", "name": "Wells Fargo Bank, National Association"},
        {"cert": "3510", "name": "Bank of America, National Association"},
        {"cert": "7213", "name": "Citibank, National Association"},
        {"cert": "628", "name": "JPMorgan Chase Bank, National Association"},
        {"cert": "6548", "name": "U.S. Bank National Association"},
        {"cert": "6384", "name": "PNC Bank, National Association"},
        {"cert": "9846", "name": "Truist Bank"},
        {"cert": "33124", "name": "Goldman Sachs Bank USA"},
        {"cert": "32992", "name": "Morgan Stanley Bank, National Association"},
        {"cert": "18409", "name": "TD Bank, National Association"},
        {"cert": "4297", "name": "Capital One, National Association"},
        {"cert": "6672", "name": "Fifth Third Bank, National Association"},
        {"cert": "57957", "name": "Citizens Bank, National Association"},
        {"cert": "57803", "name": "Ally Bank"},
        {"cert": "17534", "name": "KeyBank National Association"},

        # Regional Banks
        {"cert": "5296", "name": "Associated Bank, National Association"},
        {"cert": "4214", "name": "BOKF, National Association"},
        {"cert": "58979", "name": "BankUnited, National Association"},
        {"cert": "20234", "name": "City National Bank of Florida"},
        {"cert": "34775", "name": "EverBank, National Association"},
        {"cert": "7888", "name": "First National Bank of Pennsylvania"},
        {"cert": "3832", "name": "Old National Bank"},
        {"cert": "26881", "name": "SoFi Bank, National Association"},
        {"cert": "4988", "name": "Trustmark National Bank"},
        {"cert": "18221", "name": "Webster Bank, National Association"},
        {"cert": "33935", "name": "Wintrust Bank, National Association"},
        {"cert": "2270", "name": "Zions Bancorporation, N.A."},
        {"cert": "7551", "name": "Fulton Bank, National Association"},
        {"cert": "33555", "name": "SouthState Bank, National Association"},
        {"cert": "8273", "name": "UMB Bank, National Association"},
        {"cert": "9396", "name": "Valley National Bank"},
        {"cert": "12923", "name": "Bremer Bank, National Association"},
        {"cert": "639", "name": "The Bank of New York Mellon"},

        # Nebraska Banks
        {"cert": "10643", "name": "Dundee Bank"},
        {"cert": "19300", "name": "AMERICAN NATIONAL BANK"},
        {"cert": "20488", "name": "FIVE POINTS BANK"},
        {"cert": "5415", "name": "SECURITY FIRST BANK"},
        {"cert": "19213", "name": "SECURITY NATIONAL BANK OF OMAHA"},
        {"cert": "15545", "name": "FRONTIER BANK"},
        {"cert": "19850", "name": "WEST GATE BANK"},
        {"cert": "34363", "name": "CORE BANK"},
        {"cert": "13868", "name": "FIRST STATE BANK NEBRASKA"},
        {"cert": "58727", "name": "ACCESS BANK"},
        {"cert": "14264", "name": "CORNHUSKER BANK"},
        {"cert": "33450", "name": "ARBOR BANK"},
        {"cert": "12241", "name": "WASHINGTON COUNTY BANK"},
        {"cert": "33380", "name": "ENTERPRISE BANK"},
        {"cert": "12493", "name": "PREMIER BANK NATIONAL ASSOCIATION"},
        {"cert": "19742", "name": "FIRST WESTROADS BANK, INC."}
    ]

    start_date = '20080101'
    end_date = '20240630'

    # Process data for each bank
    all_data = []
    for bank in bank_details:
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
                    {**item['data'], 
                     'bank': bank['name'], 
                     'abbreviated_name': metrics.bank_name_mapping.get(bank['name'], bank['name'])} 
                    for item in data['data'] 
                    if isinstance(item, dict) and 'data' in item
                ]
                all_data.extend(bank_data)
        except Exception as e:
            print(f"Error processing bank {bank['name']}: {str(e)}")
            continue

    if not all_data:
        print("No data retrieved.")
        return pd.DataFrame()

    df = processor.process_raw_data(all_data)

    if df.empty:
        print("Dataframe is empty.")
        return pd.DataFrame()

    df = processor.calculate_ecl_metrics(df)
    df = processor.calculate_stress_period_changes(df)
    df = processor.calculate_predicted_ecl_and_error(df)

    return df

# Disable SSL warning for FDIC API
warnings.filterwarnings('ignore', message='.*Unverified HTTPS.*')

# Initialize data extraction and processing
processor = ECLDataProcessor()

# Get processed data
df = get_data()

# Create Dash app with modern theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
server = app.server

# Initialize dashboard
if not df.empty:
    dashboard = ECLDashboard(df, app)
else:
    print("Dataframe is empty. The dashboard will not be initialized.")

if __name__ == "__main__":
    app.run_server(debug=False)
