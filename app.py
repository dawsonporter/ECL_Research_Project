import warnings
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

BASE_URL = "https://banks.data.fdic.gov/api"

class ECLMetrics:
    def __init__(self):
        # Core model features for ECL prediction
        self.prediction_features = [
            'LNLSGR',   # Total Loans
            'RBCT1J',   # Tier 1 Capital
            'P3ASSET',  # Past Due 30-89 Days
            'P9ASSET',  # Past Due 90+ Days
            'NTLNLSQ',  # Quarterly Net Charge-Offs
            'ROA',      # Return on Assets
            'NIMY',     # Net Interest Margin
            'EEFFR',    # Efficiency Ratio
            'RBC1AAJ'   # Leverage Ratio
        ]

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
                'name': 'Global Financial Crisis',
                'stress_indicators': {
                    'gdp_decline': -2.5,
                    'unemployment_peak': 10.0,
                    'house_price_decline': -18.0
                }
            },
            'covid': {
                'start': '20200101',
                'end': '20211231',
                'name': 'COVID-19 Crisis',
                'stress_indicators': {
                    'gdp_decline': -3.5,
                    'unemployment_peak': 14.7,
                    'business_closure_rate': 22.0
                }
            }
        }
        
        # Additional stress metrics for ECL modeling
        self.stress_metrics = {
            'gdp_growth': [-2.5, -3.5, 2.0],  # Historical GDP changes during stress
            'unemployment': [10.0, 14.7, 8.0], # Peak unemployment rates
            'interest_rates': [0.25, 0.25, 2.5], # Federal funds rates
            'market_volatility': [80.0, 65.0, 35.0] # VIX index levels
        }
        
        # Define required columns for each hypothesis
        self.required_columns = {
            'h1': ['lnrecons', 'lnrenres', 'lnlsgr', 'lnatres'],
            'h2': ['lnconoth', 'lnlsgr', 'lnatres'],
            'h3': ['rbct1j', 'lnlsgr', 'lnatres'],
            'h4': self.prediction_features + ['lnatres']  # Updated for new prediction model
        }

        # Define bank groups
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

        # Updated bank name mapping with consistent abbreviations
        self.bank_name_mapping = {
            "Wells Fargo Bank, National Association": "Wells Fargo",
            "Bank of America, National Association": "Bank of America",
            "Citibank, National Association": "Citibank",
            "JPMorgan Chase Bank, National Association": "JPMorgan Chase",
            "U.S. Bank National Association": "U.S. Bank",
            "PNC Bank, National Association": "PNC Bank",
            "Truist Bank": "Truist",
            "Goldman Sachs Bank USA": "Goldman Sachs",
            "Morgan Stanley Bank, National Association": "Morgan Stanley",
            "TD Bank, National Association": "TD Bank",
            "Capital One, National Association": "Capital One",
            "Fifth Third Bank, National Association": "Fifth Third",
            "Citizens Bank, National Association": "Citizens",
            "Ally Bank": "Ally",
            "KeyBank National Association": "KeyBank",
            "Associated Bank, National Association": "Associated",
            "BOKF, National Association": "BOKF",
            "BankUnited, National Association": "BankUnited",
            "City National Bank of Florida": "City National FL",
            "EverBank, National Association": "EverBank",
            "First National Bank of Pennsylvania": "First National PA",
            "Old National Bank": "Old National",
            "SoFi Bank, National Association": "SoFi",
            "Trustmark National Bank": "Trustmark",
            "Webster Bank, National Association": "Webster",
            "Wintrust Bank, National Association": "Wintrust",
            "Zions Bancorporation, N.A.": "Zions",
            "Fulton Bank, National Association": "Fulton",
            "SouthState Bank, National Association": "SouthState",
            "UMB Bank, National Association": "UMB",
            "Valley National Bank": "Valley",
            "Bremer Bank, National Association": "Bremer",
            "The Bank of New York Mellon": "BNY Mellon",
            "Dundee Bank": "Dundee",
            "AMERICAN NATIONAL BANK": "American National",
            "FIVE POINTS BANK": "Five Points",
            "SECURITY FIRST BANK": "Security First",
            "SECURITY NATIONAL BANK OF OMAHA": "Security National",
            "FRONTIER BANK": "Frontier",
            "WEST GATE BANK": "West Gate",
            "CORE BANK": "Core",
            "FIRST STATE BANK NEBRASKA": "First State",
            "ACCESS BANK": "Access",
            "CORNHUSKER BANK": "Cornhusker",
            "ARBOR BANK": "Arbor",
            "WASHINGTON COUNTY BANK": "Washington County",
            "ENTERPRISE BANK": "Enterprise",
            "PREMIER BANK NATIONAL ASSOCIATION": "Premier",
            "FIRST WESTROADS BANK, INC.": "First Westroads"
        }
        
class ECLDataProcessor:
    def __init__(self):
        self.metrics = ECLMetrics()
        self.scaler = StandardScaler()
        
    def process_raw_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """Convert raw API data into structured DataFrame with enhanced processing"""
        if not raw_data:
            return pd.DataFrame()
            
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(raw_data)
        
        # Convert column names to lowercase
        df.rename(columns=str.lower, inplace=True)
        
        # Convert date format
        if 'repdte' in df.columns:
            df['date'] = pd.to_datetime(df['repdte'], format='%Y%m%d')
            df['year'] = df['date'].dt.year
            df['quarter'] = df['date'].dt.quarter
        
        # Convert numeric columns
        non_numeric_cols = ['cert', 'repdte', 'date', 'bank', 'abbreviated_name', 'year', 'quarter']
        numeric_columns = [col for col in df.columns if col not in non_numeric_cols]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add bank type classification
        df['bank_type'] = df['bank'].apply(self.classify_bank_type)
        
        # Add abbreviated names
        df['abbreviated_name'] = df['bank'].map(self.metrics.bank_name_mapping)
        
        # Add asset size classification
        df['asset_size_class'] = pd.qcut(df['asset'], 
                                       q=4, 
                                       labels=['Small', 'Medium', 'Large', 'Very Large'])
        
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
        """Calculate core ECL metrics with enhanced ratio calculations"""
        df = df.copy()
        
        try:
            # Calculate key ratios only if required columns exist
            if self.validate_required_columns(df, ['lnrecons', 'lnrenres', 'lnlsgr']):
                # CRE concentration with handling for zero denominators
                df['cre_concentration'] = np.where(
                    df['lnlsgr'] > 0,
                    (df['lnrecons'].fillna(0) + df['lnrenres'].fillna(0)) / df['lnlsgr'] * 100,
                    0
                )
                
            if self.validate_required_columns(df, ['lnconoth', 'lnlsgr']):
                # Consumer loan ratio with handling for zero denominators
                df['consumer_loan_ratio'] = np.where(
                    df['lnlsgr'] > 0,
                    df['lnconoth'].fillna(0) / df['lnlsgr'] * 100,
                    0
                )
                
            if self.validate_required_columns(df, ['lnatres', 'lnlsgr']):
                # ECL coverage with handling for zero denominators
                df['ecl_coverage'] = np.where(
                    df['lnlsgr'] > 0,
                    df['lnatres'].fillna(0) / df['lnlsgr'] * 100,
                    0
                )
                
            if self.validate_required_columns(df, ['ntlnlsq', 'lnlsgr']):
                # Net charge-off rate with handling for zero denominators
                df['nco_rate'] = np.where(
                    df['lnlsgr'] > 0,
                    (df['ntlnlsq'].fillna(0) * 4) / df['lnlsgr'] * 100,
                    0
                )
                
            if self.validate_required_columns(df, ['rbct1j', 'lnlsgr']):
                # Tier 1 ratio with handling for zero denominators
                df['tier1_ratio'] = np.where(
                    df['lnlsgr'] > 0,
                    df['rbct1j'].fillna(0) / df['lnlsgr'] * 100,
                    0
                )
            
            # Additional risk metrics
            if self.validate_required_columns(df, ['p3asset', 'p9asset', 'asset']):
                # Past due ratio
                df['past_due_ratio'] = np.where(
                    df['asset'] > 0,
                    (df['p3asset'].fillna(0) + df['p9asset'].fillna(0)) / df['asset'] * 100,
                    0
                )
            
            # Asset quality indicator
            if self.validate_required_columns(df, ['lnatres', 'p9asset']):
                df['coverage_ratio'] = np.where(
                    df['p9asset'] > 0,
                    df['lnatres'].fillna(0) / df['p9asset'].fillna(0) * 100,
                    0
                )
            
            # Liquidity measure
            if self.validate_required_columns(df, ['dep', 'asset']):
                df['deposit_ratio'] = np.where(
                    df['asset'] > 0,
                    df['dep'].fillna(0) / df['asset'] * 100,
                    0
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
        """Calculate metric changes during stress periods with improved handling of outliers"""
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
                    
                    # Calculate changes with outlier handling
                    changes = df.loc[period_mask, metric].groupby(df.loc[period_mask, 'cert']).pct_change() * 100
                    
                    # Remove extreme outliers (beyond 3 std dev)
                    mean_change = changes.mean()
                    std_change = changes.std()
                    changes = np.clip(changes, 
                                    mean_change - 3 * std_change,
                                    mean_change + 3 * std_change)
                    
                    df.loc[period_mask, change_col] = changes
                    
                    # Add rolling averages for trend analysis
                    df.loc[period_mask, f'{change_col}_3m_avg'] = (
                        df.loc[period_mask, change_col].groupby(df.loc[period_mask, 'cert'])
                        .rolling(window=3, min_periods=1)
                        .mean()
                        .reset_index(0, drop=True)
                    )
        
        return df

    def prepare_prediction_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features for ECL prediction model"""
        # Create lowercase version of prediction features
        lowercase_features = [feat.lower() for feat in self.metrics.prediction_features]
        
        # Select relevant features
        features = df[lowercase_features].copy()
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Scale features
        scaled_features = pd.DataFrame(
            self.scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        
        # Create target variable (ECL)
        target = df['lnatres'].copy()
        
        return scaled_features, target

    def calculate_predicted_ecl_and_error(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate predicted ECL using advanced modeling techniques"""
        df = df.copy()
        try:
            # Prepare features and target
            features, target = self.prepare_prediction_features(df)
            
            # Split data for model training
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Train Random Forest model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Make predictions
            df['predicted_ecl'] = rf_model.predict(features)
            
            # Calculate prediction errors
            df['prediction_error'] = (
                (df['predicted_ecl'] - df['lnatres']) / df['lnatres'] * 100
            )
            
            # Calculate model performance metrics
            r2_train = r2_score(y_train, rf_model.predict(X_train))
            r2_test = r2_score(y_test, rf_model.predict(X_test))
            
            # Store model performance metrics
            df['model_r2_train'] = r2_train
            df['model_r2_test'] = r2_test
            
            # Feature importance analysis
            importance_df = pd.DataFrame({
                'feature': self.metrics.prediction_features,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Store top 3 important features
            df['top_predictors'] = ', '.join(importance_df['feature'].head(3).tolist())
            
        except Exception as e:
            print(f"Error calculating predicted ECL and error: {str(e)}")
            
        return df
    
class HypothesisTester:
    def __init__(self):
        self.metrics = ECLMetrics()
        self.significance_level = 0.05
        self.min_sample_size = 30  # Minimum sample size for reliable testing

    def test_hypothesis_1(self, df: pd.DataFrame, period: str) -> Dict:
        """Test if banks with high CRE concentration experience larger ECL increases with enhanced analysis"""
        try:
            period_info = self.metrics.stress_periods[period]
            
            # Filter for stress period
            stress_df = df[
                (df['date'] >= pd.to_datetime(period_info['start'])) &
                (df['date'] <= pd.to_datetime(period_info['end']))
            ].copy()
            
            # Remove invalid values and outliers
            stress_df = self._clean_data(
                stress_df, 
                ['cre_concentration', f'ecl_coverage_change_{period}']
            )
            
            if len(stress_df) < self.min_sample_size:
                return {'error': 'Insufficient data for reliable analysis'}
            
            # Create quartiles based on CRE concentration
            stress_df['cre_quartile'] = pd.qcut(
                stress_df['cre_concentration'], 
                q=4, 
                labels=['Q1', 'Q2', 'Q3', 'Q4']
            )
            
            # Calculate differences between top and bottom quartiles
            top_quartile = stress_df[stress_df['cre_quartile'] == 'Q4'][f'ecl_coverage_change_{period}']
            bottom_quartile = stress_df[stress_df['cre_quartile'] == 'Q1'][f'ecl_coverage_change_{period}']
            
            # Perform statistical tests
            mw_stat, mw_pvalue = stats.mannwhitneyu(
                top_quartile.dropna(),
                bottom_quartile.dropna(),
                alternative='greater'
            )
            
            # Calculate effect size (Cohen's d)
            effect_size = (top_quartile.mean() - bottom_quartile.mean()) / np.sqrt(
                (top_quartile.var() + bottom_quartile.var()) / 2
            )
            
            # Additional statistical analysis
            ttest_stat, ttest_pvalue = stats.ttest_ind(
                top_quartile.dropna(),
                bottom_quartile.dropna()
            )
            
            # Run regression analysis
            X = sm.add_constant(stress_df['cre_concentration'])
            y = stress_df[f'ecl_coverage_change_{period}']
            model = sm.OLS(y, X).fit()
            
            return {
                'summary_statistics': {
                    'top_quartile_mean': top_quartile.mean(),
                    'bottom_quartile_mean': bottom_quartile.mean(),
                    'median_difference': top_quartile.median() - bottom_quartile.median(),
                    'effect_size': effect_size,
                    'sample_size': len(top_quartile) + len(bottom_quartile)
                },
                'statistical_tests': {
                    'mann_whitney_u': {
                        'statistic': mw_stat,
                        'p_value': mw_pvalue
                    },
                    't_test': {
                        'statistic': ttest_stat,
                        'p_value': ttest_pvalue
                    }
                },
                'regression_analysis': {
                    'coefficient': model.params[1],
                    'r_squared': model.rsquared,
                    'p_value': model.pvalues[1]
                },
                'hypothesis_supported': (
                    mw_pvalue < self.significance_level and
                    effect_size > 0.5
                )
            }
            
        except Exception as e:
            print(f"Error testing hypothesis 1: {str(e)}")
            return {'error': str(e)}

    def test_hypothesis_2(self, df: pd.DataFrame, period: str) -> Dict:
        """Test consumer loan portfolio sensitivity with enhanced analysis"""
        try:
            period_info = self.metrics.stress_periods[period]
            
            # Filter for stress period
            stress_df = df[
                (df['date'] >= pd.to_datetime(period_info['start'])) &
                (df['date'] <= pd.to_datetime(period_info['end']))
            ].copy()
            
            # Remove invalid values and outliers
            stress_df = self._clean_data(
                stress_df, 
                ['consumer_loan_ratio', f'ecl_coverage_change_{period}']
            )
            
            if len(stress_df) < self.min_sample_size:
                return {'error': 'Insufficient data for reliable analysis'}

            # Calculate median consumer loan ratio
            median_consumer_ratio = stress_df['consumer_loan_ratio'].median()

            # Split banks by consumer loan concentration
            high_consumer = stress_df[stress_df['consumer_loan_ratio'] > median_consumer_ratio]
            low_consumer = stress_df[stress_df['consumer_loan_ratio'] <= median_consumer_ratio]

            # Calculate ECL changes
            metric = f'ecl_coverage_change_{period}'
            high_changes = high_consumer[metric].dropna()
            low_changes = low_consumer[metric].dropna()

            # Perform statistical tests
            mw_stat, mw_pvalue = stats.mannwhitneyu(
                high_changes,
                low_changes,
                alternative='greater'
            )
            
            # Calculate effect size
            effect_size = (high_changes.mean() - low_changes.mean()) / np.sqrt(
                (high_changes.var() + low_changes.var()) / 2
            )
            
            # Run regression analysis
            X = sm.add_constant(stress_df['consumer_loan_ratio'])
            y = stress_df[metric]
            model = sm.OLS(y, X).fit()
            
            # Time series analysis
            high_consumer_ts = (
                high_consumer.groupby('date')[metric]
                .mean()
                .rolling(window=3)
                .mean()
            )
            
            low_consumer_ts = (
                low_consumer.groupby('date')[metric]
                .mean()
                .rolling(window=3)
                .mean()
            )

            return {
                'summary_statistics': {
                    'high_consumer_mean': high_changes.mean(),
                    'low_consumer_mean': low_changes.mean(),
                    'median_difference': high_changes.median() - low_changes.median(),
                    'effect_size': effect_size,
                    'sample_size': {
                        'high_consumer': len(high_changes),
                        'low_consumer': len(low_changes)
                    }
                },
                'statistical_tests': {
                    'mann_whitney_u': {
                        'statistic': mw_stat,
                        'p_value': mw_pvalue
                    }
                },
                'regression_analysis': {
                    'coefficient': model.params[1],
                    'r_squared': model.rsquared,
                    'p_value': model.pvalues[1]
                },
                'time_series_analysis': {
                    'high_consumer_trend': high_consumer_ts.to_dict(),
                    'low_consumer_trend': low_consumer_ts.to_dict()
                },
                'hypothesis_supported': (
                    mw_pvalue < self.significance_level and
                    effect_size > 0.3
                )
            }
            
        except Exception as e:
            print(f"Error testing hypothesis 2: {str(e)}")
            return {'error': str(e)}

    def test_hypothesis_3(self, df: pd.DataFrame, period: str) -> Dict:
        """Test if higher capital ratios lead to smaller ECL increases with enhanced analysis"""
        try:
            period_info = self.metrics.stress_periods[period]
            
            # Filter for stress period
            stress_df = df[
                (df['date'] >= pd.to_datetime(period_info['start'])) &
                (df['date'] <= pd.to_datetime(period_info['end']))
            ].copy()
            
            # Remove invalid values and outliers
            stress_df = self._clean_data(
                stress_df, 
                ['tier1_ratio', f'ecl_coverage_change_{period}']
            )
            
            if len(stress_df) < self.min_sample_size:
                return {'error': 'Insufficient data for reliable analysis'}

            # Calculate median Tier 1 ratio
            median_tier1 = stress_df['tier1_ratio'].median()
            
            # Split banks by Tier 1 ratio
            high_capital = stress_df[stress_df['tier1_ratio'] > median_tier1]
            low_capital = stress_df[stress_df['tier1_ratio'] <= median_tier1]
            
            # Calculate ECL changes
            metric = f'ecl_coverage_change_{period}'
            high_changes = high_capital[metric].dropna()
            low_changes = low_capital[metric].dropna()
            
            # Perform statistical tests
            mw_stat, mw_pvalue = stats.mannwhitneyu(
                high_changes,
                low_changes,
                alternative='less'
            )
            
            # Calculate effect size
            effect_size = (high_changes.mean() - low_changes.mean()) / np.sqrt(
                (high_changes.var() + low_changes.var()) / 2
            )
            
            # Run regression analysis
            X = sm.add_constant(stress_df['tier1_ratio'])
            y = stress_df[metric]
            model = sm.OLS(y, X).fit()
            
            # Additional analysis by bank type
            bank_type_analysis = {}
            for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
                type_data = stress_df[stress_df['bank_type'] == bank_type]
                if len(type_data) > 0:
                    bank_type_analysis[bank_type] = {
                        'mean_tier1': type_data['tier1_ratio'].mean(),
                        'mean_ecl_change': type_data[metric].mean(),
                        'correlation': type_data['tier1_ratio'].corr(type_data[metric])
                    }

            return {
                'summary_statistics': {
                    'high_capital_mean': high_changes.mean(),
                    'low_capital_mean': low_changes.mean(),
                    'median_difference': high_changes.median() - low_changes.median(),
                    'effect_size': effect_size,
                    'sample_size': {
                        'high_capital': len(high_changes),
                        'low_capital': len(low_changes)
                    }
                },
                'statistical_tests': {
                    'mann_whitney_u': {
                        'statistic': mw_stat,
                        'p_value': mw_pvalue
                    }
                },
                'regression_analysis': {
                    'coefficient': model.params[1],
                    'r_squared': model.rsquared,
                    'p_value': model.pvalues[1]
                },
                'bank_type_analysis': bank_type_analysis,
                'hypothesis_supported': (
                    mw_pvalue < self.significance_level and
                    effect_size < -0.3
                )
            }
            
        except Exception as e:
            print(f"Error testing hypothesis 3: {str(e)}")
            return {'error': str(e)}

    def test_hypothesis_4(self, df: pd.DataFrame, period: str) -> Dict:
        """Test ECL prediction model accuracy with comprehensive validation"""
        try:
            period_info = self.metrics.stress_periods[period]
            
            # Filter for stress period
            stress_df = df[
                (df['date'] >= pd.to_datetime(period_info['start'])) &
                (df['date'] <= pd.to_datetime(period_info['end']))
            ].copy()
            
            # Remove invalid values
            stress_df = self._clean_data(
                stress_df, 
                ['lnatres', 'predicted_ecl', 'prediction_error']
            )
            
            if len(stress_df) < self.min_sample_size:
                return {'error': 'Insufficient data for reliable analysis'}
            
            # Calculate prediction accuracy metrics
            actual_ecl = stress_df['lnatres']
            predicted_ecl = stress_df['predicted_ecl']
            prediction_errors = stress_df['prediction_error'].abs()
            
            # Calculate R-squared
            r2 = r2_score(actual_ecl, predicted_ecl)
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(actual_ecl, predicted_ecl))
            
            # Calculate accuracy by bank type
            bank_type_analysis = {}
            for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
                type_data = stress_df[stress_df['bank_type'] == bank_type]
                if len(type_data) > 0:
                    bank_type_analysis[bank_type] = {
                        'mean_error': type_data['prediction_error'].mean(),
                        'median_error': type_data['prediction_error'].median(),
                        'rmse': np.sqrt(mean_squared_error(
                            type_data['lnatres'],
                            type_data['predicted_ecl']
                        )),
                        'r2': r2_score(
                            type_data['lnatres'],
                            type_data['predicted_ecl']
                        )
                    }

            return {
                'model_performance': {
                    'r_squared': r2,
                    'rmse': rmse,
                    'mean_absolute_error': prediction_errors.mean(),
                    'median_absolute_error': prediction_errors.median(),
                    'prediction_within_10_percent': (prediction_errors <= 10).mean() * 100
                },
                'error_distribution': {
                    'std_dev': prediction_errors.std(),
                    'skewness': prediction_errors.skew(),
                    'kurtosis': prediction_errors.kurtosis(),
                    'percentiles': {
                        '25th': prediction_errors.quantile(0.25),
                        '50th': prediction_errors.quantile(0.50),
                        '75th': prediction_errors.quantile(0.75),
                        '90th': prediction_errors.quantile(0.90)
                    }
                },
                'bank_type_analysis': bank_type_analysis,
                'time_series_analysis': {
                    'error_trend': stress_df.groupby('date')['prediction_error'].mean().to_dict(),
                    'accuracy_trend': stress_df.groupby('date').apply(
                        lambda x: (x['prediction_error'].abs() <= 10).mean() * 100
                    ).to_dict()
                },
                'hypothesis_supported': (
                    r2 >= 0.7 and
                    (prediction_errors <= 10).mean() >= 0.9
                )
            }
            
        except Exception as e:
            print(f"Error testing hypothesis 4: {str(e)}")
            return {'error': str(e)}

    def _clean_data(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Clean data by removing invalid values and outliers"""
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)
        
        # Remove outliers (beyond 3 standard deviations)
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[
                (df[col] >= mean - 3 * std) &
                (df[col] <= mean + 3 * std)
            ]
        
        return df

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
            'nebraska': '#bc5090',
            'trend': '#ff6361'
        }
        self.layout_defaults = dict(
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font=dict(color='#333333'),
            height=400,  # Reduced height for better layout
            margin=dict(l=50, r=20, t=50, b=50)
        )

    def create_hypothesis1_visualization(self, df: pd.DataFrame, period: str) -> Dict[str, go.Figure]:
        """Create multiple visualizations for CRE concentration hypothesis"""
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
        
        figures = {}
        
        # Figure 1: Box Plot by Bank Type
        box_fig = go.Figure()
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            box_fig.add_trace(
                go.Box(
                    x=bank_data['cre_quartile'],
                    y=bank_data[f'ecl_coverage_change_{period}'],
                    name=bank_type,
                    marker_color=self.color_scheme[bank_type.split()[0].lower()]
                )
            )
        box_fig.update_layout(
            title="ECL Changes by CRE Concentration Quartile and Bank Type",
            xaxis_title="CRE Concentration Quartile",
            yaxis_title="ECL Change (%)",
            **self.layout_defaults
        )
        figures['box_plot'] = box_fig
        
        # Figure 2: Time Series Trend
        time_fig = go.Figure()
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type].sort_values('date')
            time_fig.add_trace(
                go.Scatter(
                    x=bank_data['date'],
                    y=bank_data[f'ecl_coverage_change_{period}'],
                    name=bank_type,
                    mode='lines+markers',
                    line=dict(color=self.color_scheme[bank_type.split()[0].lower()])
                )
            )
        time_fig.update_layout(
            title="Time Series of ECL Changes by Bank Type",
            xaxis_title="Date",
            yaxis_title="ECL Change (%)",
            **self.layout_defaults
        )
        figures['time_series'] = time_fig
        
        # Figure 3: Scatter Plot with Trend Lines
        scatter_fig = go.Figure()
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            
            # Add scatter points
            scatter_fig.add_trace(
                go.Scatter(
                    x=bank_data['cre_concentration'],
                    y=bank_data[f'ecl_coverage_change_{period}'],
                    mode='markers',
                    name=f'{bank_type} Data',
                    marker=dict(
                        color=self.color_scheme[bank_type.split()[0].lower()],
                        size=8,
                        opacity=0.6
                    )
                )
            )
            
            # Add trend line
            z = np.polyfit(bank_data['cre_concentration'], 
                          bank_data[f'ecl_coverage_change_{period}'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(bank_data['cre_concentration'].min(), 
                                bank_data['cre_concentration'].max(), 100)
            scatter_fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name=f'{bank_type} Trend',
                    line=dict(
                        color=self.color_scheme[bank_type.split()[0].lower()],
                        dash='dash'
                    )
                )
            )
        
        scatter_fig.update_layout(
            title="ECL Change vs CRE Concentration by Bank Type",
            xaxis_title="CRE Concentration (%)",
            yaxis_title="ECL Change (%)",
            **self.layout_defaults
        )
        figures['scatter_plot'] = scatter_fig
        
        # Figure 4: Distribution Analysis
        dist_fig = go.Figure()
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            dist_fig.add_trace(
                go.Histogram(
                    x=bank_data['cre_concentration'],
                    name=bank_type,
                    marker_color=self.color_scheme[bank_type.split()[0].lower()],
                    opacity=0.7,
                    nbinsx=30
                )
            )
        dist_fig.update_layout(
            title="CRE Concentration Distribution by Bank Type",
            xaxis_title="CRE Concentration (%)",
            yaxis_title="Count",
            barmode='overlay',
            **self.layout_defaults
        )
        figures['distribution'] = dist_fig
        
        return figures

    def create_hypothesis2_visualization(self, df: pd.DataFrame, period: str) -> Dict[str, go.Figure]:
        """Create multiple visualizations for consumer loan sensitivity hypothesis"""
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
        stress_df['consumer_group'] = np.where(
            stress_df['consumer_loan_ratio'] > median_consumer_ratio,
            'High Consumer',
            'Low Consumer'
        )
        
        figures = {}
        
        # Figure 1: Box Plot by Bank Type and Consumer Group
        box_fig = go.Figure()
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            for consumer_group in ['High Consumer', 'Low Consumer']:
                group_data = stress_df[
                    (stress_df['bank_type'] == bank_type) & 
                    (stress_df['consumer_group'] == consumer_group)
                ]
                box_fig.add_trace(
                    go.Box(
                        x=[f"{bank_type}<br>{consumer_group}"],
                        y=group_data[f'ecl_coverage_change_{period}'],
                        name=f"{bank_type} - {consumer_group}",
                        marker_color=self.color_scheme[bank_type.split()[0].lower()],
                        opacity=0.7 if consumer_group == 'Low Consumer' else 1
                    )
                )
        box_fig.update_layout(
            title="ECL Changes by Bank Type and Consumer Loan Concentration",
            xaxis_title="Bank Type and Consumer Loan Group",
            yaxis_title="ECL Change (%)",
            showlegend=True,
            **self.layout_defaults
        )
        figures['box_plot'] = box_fig

        # Figure 2: Time Series Analysis
        time_fig = go.Figure()
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            for consumer_group in ['High Consumer', 'Low Consumer']:
                group_data = stress_df[
                    (stress_df['bank_type'] == bank_type) & 
                    (stress_df['consumer_group'] == consumer_group)
                ].sort_values('date')
                
                # Calculate moving average
                group_data['ma'] = group_data[f'ecl_coverage_change_{period}'].rolling(window=3).mean()
                
                time_fig.add_trace(
                    go.Scatter(
                        x=group_data['date'],
                        y=group_data['ma'],
                        name=f"{bank_type} - {consumer_group}",
                        mode='lines',
                        line=dict(
                            color=self.color_scheme[bank_type.split()[0].lower()],
                            dash='dash' if consumer_group == 'Low Consumer' else 'solid'
                        )
                    )
                )
        time_fig.update_layout(
            title="Time Series of ECL Changes by Bank Type and Consumer Loan Group",
            xaxis_title="Date",
            yaxis_title="ECL Change (%) - 3-Month Moving Average",
            **self.layout_defaults
        )
        figures['time_series'] = time_fig
        
        # Figure 3: Scatter Plot with Regression Lines
        scatter_fig = go.Figure()
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            
            # Add scatter points
            scatter_fig.add_trace(
                go.Scatter(
                    x=bank_data['consumer_loan_ratio'],
                    y=bank_data[f'ecl_coverage_change_{period}'],
                    mode='markers',
                    name=f'{bank_type}',
                    marker=dict(
                        color=self.color_scheme[bank_type.split()[0].lower()],
                        size=8,
                        opacity=0.6
                    )
                )
            )
            
            # Add regression line
            z = np.polyfit(bank_data['consumer_loan_ratio'], 
                          bank_data[f'ecl_coverage_change_{period}'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(bank_data['consumer_loan_ratio'].min(), 
                                bank_data['consumer_loan_ratio'].max(), 100)
            scatter_fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name=f'{bank_type} Trend',
                    line=dict(
                        color=self.color_scheme[bank_type.split()[0].lower()],
                        dash='dash'
                    )
                )
            )
            
        scatter_fig.update_layout(
            title="ECL Change vs Consumer Loan Ratio",
            xaxis_title="Consumer Loan Ratio (%)",
            yaxis_title="ECL Change (%)",
            **self.layout_defaults
        )
        figures['scatter_plot'] = scatter_fig
        
        # Figure 4: Risk Profile Analysis
        risk_fig = go.Figure()
        
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            
            # Calculate risk metrics
            risk_data = bank_data.groupby('consumer_group').agg({
                f'ecl_coverage_change_{period}': ['mean', 'std'],
                'tier1_ratio': 'mean'
            }).reset_index()
            
            risk_fig.add_trace(
                go.Scatter(
                    x=risk_data['consumer_group'],
                    y=risk_data[f'ecl_coverage_change_{period}']['mean'],
                    mode='markers+text',
                    name=bank_type,
                    marker=dict(
                        color=self.color_scheme[bank_type.split()[0].lower()],
                        size=risk_data['tier1_ratio']['mean'] * 2,  # Size based on tier 1 ratio
                        sizemode='diameter'
                    ),
                    text=bank_type,
                    textposition='top center'
                )
            )
            
        risk_fig.update_layout(
            title="Risk Profile by Bank Type and Consumer Loan Exposure",
            xaxis_title="Consumer Loan Group",
            yaxis_title="Mean ECL Change (%)",
            **self.layout_defaults
        )
        figures['risk_profile'] = risk_fig
        
        return figures

    def create_hypothesis3_visualization(self, df: pd.DataFrame, period: str) -> Dict[str, go.Figure]:
        """Create multiple visualizations for capital structure hypothesis"""
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
        
        # Calculate median tier1 ratio
        median_tier1 = stress_df['tier1_ratio'].median()
        stress_df['capital_group'] = np.where(
            stress_df['tier1_ratio'] > median_tier1,
            'High Capital',
            'Low Capital'
        )
        
        figures = {}
        
        # Figure 1: Capital Buffer Analysis
        buffer_fig = go.Figure()
        
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            
            buffer_fig.add_trace(
                go.Box(
                    x=bank_data['capital_group'],
                    y=bank_data[f'ecl_coverage_change_{period}'],
                    name=bank_type,
                    marker_color=self.color_scheme[bank_type.split()[0].lower()]
                )
            )
            
        buffer_fig.update_layout(
            title="Capital Buffer Impact on ECL Changes",
            xaxis_title="Capital Level",
            yaxis_title="ECL Change (%)",
            **self.layout_defaults
        )
        figures['capital_buffer'] = buffer_fig
        
        # Figure 2: Time Series of Capital Adequacy
        time_fig = go.Figure()
        
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type].sort_values('date')
            
            # Calculate moving averages
            bank_data['ma_tier1'] = bank_data['tier1_ratio'].rolling(window=3).mean()
            bank_data['ma_ecl'] = bank_data[f'ecl_coverage_change_{period}'].rolling(window=3).mean()
            
            # Create dual-axis plot
            time_fig.add_trace(
                go.Scatter(
                    x=bank_data['date'],
                    y=bank_data['ma_tier1'],
                    name=f"{bank_type} - Tier 1",
                    mode='lines',
                    line=dict(color=self.color_scheme[bank_type.split()[0].lower()]),
                    yaxis='y'
                )
            )
            
            time_fig.add_trace(
                go.Scatter(
                    x=bank_data['date'],
                    y=bank_data['ma_ecl'],
                    name=f"{bank_type} - ECL",
                    mode='lines',
                    line=dict(
                        color=self.color_scheme[bank_type.split()[0].lower()],
                        dash='dash'
                    ),
                    yaxis='y2'
                )
            )
            
        time_fig.update_layout(
            title="Time Series of Capital Adequacy and ECL Changes",
            xaxis_title="Date",
            yaxis_title="Tier 1 Ratio (%)",
            yaxis2=dict(
                title="ECL Change (%)",
                overlaying='y',
                side='right'
            ),
            **self.layout_defaults
        )
        figures['time_series'] = time_fig
        
        # Figure 3: Capital Efficiency Analysis
        efficiency_fig = go.Figure()
        
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            
            efficiency_fig.add_trace(
                go.Scatter(
                    x=bank_data['tier1_ratio'],
                    y=bank_data[f'ecl_coverage_change_{period}'],
                    mode='markers',
                    name=bank_type,
                    marker=dict(
                        color=self.color_scheme[bank_type.split()[0].lower()],
                        size=10,
                        opacity=0.6
                    )
                )
            )
            
            # Add trend line
            z = np.polyfit(bank_data['tier1_ratio'], 
                          bank_data[f'ecl_coverage_change_{period}'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(bank_data['tier1_ratio'].min(), 
                                bank_data['tier1_ratio'].max(), 100)
            
            efficiency_fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name=f'{bank_type} Trend',
                    line=dict(
                        color=self.color_scheme[bank_type.split()[0].lower()],
                        dash='dash'
                    )
                )
            )
            
        efficiency_fig.update_layout(
            title="Capital Efficiency Analysis",
            xaxis_title="Tier 1 Ratio (%)",
            yaxis_title="ECL Change (%)",
            **self.layout_defaults
        )
        figures['efficiency'] = efficiency_fig
        
        # Figure 4: Risk-Return Analysis
        risk_return_fig = go.Figure()
        
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            
            risk_return_fig.add_trace(
                go.Scatter(
                    x=bank_data['tier1_ratio'],
                    y=bank_data['roa'],  # Return on Assets
                    mode='markers',
                    name=bank_type,
                    marker=dict(
                        color=self.color_scheme[bank_type.split()[0].lower()],
                        size=abs(bank_data[f'ecl_coverage_change_{period}']) / 2,  # Size based on ECL change
                        sizemode='diameter',
                        sizeref=2.*max(abs(stress_df[f'ecl_coverage_change_{period}']))/(40.**2),
                        sizemin=4
                    )
                )
            )
            
        risk_return_fig.update_layout(
            title="Risk-Return Profile by Capital Level",
            xaxis_title="Tier 1 Ratio (%)",
            yaxis_title="Return on Assets (%)",
            **self.layout_defaults
        )
        figures['risk_return'] = risk_return_fig
        
        return figures

    def create_hypothesis4_visualization(self, df: pd.DataFrame, period: str) -> Dict[str, go.Figure]:
        """Create multiple visualizations for ECL prediction accuracy"""
        period_info = self.metrics.stress_periods[period]
        
        # Filter and prepare data
        stress_df = df[
            (df['date'] >= pd.to_datetime(period_info['start'])) &
            (df['date'] <= pd.to_datetime(period_info['end']))
        ].copy()
        
        # Remove invalid values
        stress_df = stress_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=['lnatres', 'predicted_ecl', 'prediction_error']
        )
        
        figures = {}
        
        # Figure 1: Prediction Accuracy Overview
        accuracy_fig = go.Figure()
        
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            
            accuracy_fig.add_trace(
                go.Scatter(
                    x=bank_data['lnatres'],
                    y=bank_data['predicted_ecl'],
                    mode='markers',
                    name=bank_type,
                    marker=dict(
                        color=self.color_scheme[bank_type.split()[0].lower()],
                        size=8,
                        opacity=0.6
                    )
                )
            )
            
        # Add perfect prediction line
        max_val = max(stress_df['lnatres'].max(), stress_df['predicted_ecl'].max())
        accuracy_fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='black', dash='dash')
            )
        )
        
        accuracy_fig.update_layout(
            title="Predicted vs Actual ECL",
            xaxis_title="Actual ECL",
            yaxis_title="Predicted ECL",
            **self.layout_defaults
        )
        figures['accuracy'] = accuracy_fig
        
        # Figure 2: Error Distribution
        error_fig = go.Figure()
        
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type]
            
            error_fig.add_trace(
                go.Histogram(
                    x=bank_data['prediction_error'],
                    name=bank_type,
                    marker_color=self.color_scheme[bank_type.split()[0].lower()],
                    opacity=0.7,
                    nbinsx=30
                )
            )
            
        error_fig.update_layout(
            title="Prediction Error Distribution by Bank Type",
            xaxis_title="Prediction Error (%)",
            yaxis_title="Count",
            barmode='overlay',
            **self.layout_defaults
        )
        figures['error_distribution'] = error_fig
        
        # Figure 3: Time Series of Prediction Accuracy
        time_fig = go.Figure()
        
        for bank_type in ['National Systemic Bank', 'Regional Bank', 'Nebraska Bank']:
            bank_data = stress_df[stress_df['bank_type'] == bank_type].sort_values('date')
            
            # Calculate moving average of absolute error
            bank_data['ma_error'] = bank_data['prediction_error'].abs().rolling(window=3).mean()
            
            time_fig.add_trace(
                go.Scatter(
                    x=bank_data['date'],
                    y=bank_data['ma_error'],
                    name=bank_type,
                    mode='lines',
                    line=dict(color=self.color_scheme[bank_type.split()[0].lower()])
                )
            )
            
        time_fig.update_layout(
            title="Prediction Error Over Time",
            xaxis_title="Date",
            yaxis_title="Absolute Prediction Error (%) - 3-Month Moving Average",
            **self.layout_defaults
        )
        figures['time_series'] = time_fig
        
        # Figure 4: Feature Importance Analysis
        importance_fig = go.Figure()
        
        # Get feature importance data
        feature_importance = pd.DataFrame({
            'feature': stress_df['top_predictors'].iloc[0].split(', '),
            'importance': range(3, 0, -1)  # Importance score based on order
        })
        
        importance_fig.add_trace(
            go.Bar(
                x=feature_importance['feature'],
                y=feature_importance['importance'],
                marker_color=self.color_scheme['primary']
            )
        )
        
        importance_fig.update_layout(
            title="Top Predictive Features for ECL",
            xaxis_title="Feature",
            yaxis_title="Relative Importance",
            **self.layout_defaults
        )
        figures['feature_importance'] = importance_fig
        
        return figures

class ECLDashboard:
    def __init__(self, df: pd.DataFrame, app: dash.Dash):
        self.df = df
        self.app = app
        self.visualizer = ECLVisualizer()
        self.hypothesis_tester = HypothesisTester()
        self.metrics = ECLMetrics()
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("ECL Research Analysis Dashboard",
                           className="text-primary mb-4"),
                    html.H5("Analysis of Expected Credit Loss Determinants",
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
                                    {'label': 'H4: ECL Prediction Accuracy', 
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
                                value=['national', 'regional', 'nebraska'],
                                className="mt-2"
                            )
                        ])
                    ], className="mb-4"),

                    # Bank Categories Panel
                    dbc.Card([
                        dbc.CardHeader("Bank Categories"),
                        dbc.CardBody([
                            dbc.Tabs([
                                dbc.Tab([
                                    html.Div(id='national-bank-list')
                                ], label="National", className="mt-3"),
                                dbc.Tab([
                                    html.Div(id='regional-bank-list')
                                ], label="Regional", className="mt-3"),
                                dbc.Tab([
                                    html.Div(id='nebraska-bank-list')
                                ], label="Nebraska", className="mt-3")
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
                            # Four separate plot sections
                            dbc.Row([
                                dbc.Col([
                                    dcc.Loading(
                                        id="loading-1",
                                        type="default",
                                        children=[dcc.Graph(id='plot-1')]
                                    )
                                ], width=6),
                                dbc.Col([
                                    dcc.Loading(
                                        id="loading-2",
                                        type="default",
                                        children=[dcc.Graph(id='plot-2')]
                                    )
                                ], width=6)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Loading(
                                        id="loading-3",
                                        type="default",
                                        children=[dcc.Graph(id='plot-3')]
                                    )
                                ], width=6),
                                dbc.Col([
                                    dcc.Loading(
                                        id="loading-4",
                                        type="default",
                                        children=[dcc.Graph(id='plot-4')]
                                    )
                                ], width=6)
                            ], className="mt-4")
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
                                id="loading-stats",
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
                                id="loading-metrics",
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
            [Output('plot-1', 'figure'),
             Output('plot-2', 'figure'),
             Output('plot-3', 'figure'),
             Output('plot-4', 'figure'),
             Output('statistical-results', 'children'),
             Output('supporting-metrics', 'children'),
             Output('hypothesis-description', 'children'),
             Output('national-bank-list', 'children'),
             Output('regional-bank-list', 'children'),
             Output('nebraska-bank-list', 'children')],
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

            # Generate visualizations based on hypothesis
            if hypothesis == 'h1':
                figures = self.visualizer.create_hypothesis1_visualization(
                    filtered_df, stress_period
                )
                results = self.hypothesis_tester.test_hypothesis_1(
                    filtered_df, stress_period
                )
                description = self.get_hypothesis_description('h1')

            elif hypothesis == 'h2':
                figures = self.visualizer.create_hypothesis2_visualization(
                    filtered_df, stress_period
                )
                results = self.hypothesis_tester.test_hypothesis_2(
                    filtered_df, stress_period
                )
                description = self.get_hypothesis_description('h2')

            elif hypothesis == 'h3':
                figures = self.visualizer.create_hypothesis3_visualization(
                    filtered_df, stress_period
                )
                results = self.hypothesis_tester.test_hypothesis_3(
                    filtered_df, stress_period
                )
                description = self.get_hypothesis_description('h3')

            elif hypothesis == 'h4':
                figures = self.visualizer.create_hypothesis4_visualization(
                    filtered_df, stress_period
                )
                results = self.hypothesis_tester.test_hypothesis_4(
                    filtered_df, stress_period
                )
                description = self.get_hypothesis_description('h4')

            else:
                figures = {
                    'box_plot': go.Figure(),
                    'time_series': go.Figure(),
                    'scatter_plot': go.Figure(),
                    'distribution': go.Figure()
                }
                results = {}
                description = ""

            stats_panel = self.create_stats_panel(results)
            supporting_metrics = self.create_supporting_metrics_panel(
                filtered_df, stress_period, hypothesis
            )

            # Generate bank lists by type using abbreviated names
            national_banks = [
                dbc.ListGroupItem(
                    self.metrics.bank_name_mapping.get(bank, bank),
                    className="py-2"
                ) for bank in sorted(filtered_df[
                    filtered_df['bank_type'] == 'National Systemic Bank'
                ]['bank'].unique())
            ]
            national_list = national_banks if national_banks else [
                dbc.ListGroupItem("No national banks available")
            ]

            regional_banks = [
                dbc.ListGroupItem(
                    self.metrics.bank_name_mapping.get(bank, bank),
                    className="py-2"
                ) for bank in sorted(filtered_df[
                    filtered_df['bank_type'] == 'Regional Bank'
                ]['bank'].unique())
            ]
            regional_list = regional_banks if regional_banks else [
                dbc.ListGroupItem("No regional banks available")
            ]

            nebraska_banks = [
                dbc.ListGroupItem(
                    self.metrics.bank_name_mapping.get(bank, bank),
                    className="py-2"
                ) for bank in sorted(filtered_df[
                    filtered_df['bank_type'] == 'Nebraska Bank'
                ]['bank'].unique())
            ]
            nebraska_list = nebraska_banks if nebraska_banks else [
                dbc.ListGroupItem("No Nebraska banks available")
            ]

            return (
                figures['box_plot'], figures['time_series'], 
                figures['scatter_plot'], figures['distribution'],
                stats_panel, supporting_metrics, description,
                national_list, regional_list, nebraska_list
            )

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
        """Return detailed description of selected hypothesis"""
        descriptions = {
            'h1': """
                Hypothesis 1: CRE Concentration Impact
                Banks with high commercial real estate (CRE) concentration experience significantly larger ECL increases 
                during stress periods. We expect at least 20% larger ECL increases in the top quartile compared to the 
                bottom quartile of CRE concentration. This relationship may vary by bank type and size.
            """,
            'h2': """
                Hypothesis 2: Consumer Loan Sensitivity
                Banks with higher consumer loan concentrations show greater ECL sensitivity during stress periods,
                particularly during economic downturns affecting household income and employment. The analysis 
                considers different bank types and their varying exposure to consumer credit risk.
            """,
            'h3': """
                Hypothesis 3: Capital Structure Effect
                Banks with higher Tier 1 capital ratios experience smaller ECL increases during stress periods,
                suggesting better risk absorption capacity. We expect at least 15% smaller ECL increases for 
                well-capitalized banks, with variations across different bank types and sizes.
            """,
            'h4': """
                Hypothesis 4: ECL Prediction Accuracy
                Our machine learning model can predict ECL levels within 10% accuracy using key financial metrics
                and stress indicators. The model's performance is evaluated across different bank types and stress
                periods, with particular attention to prediction reliability during crisis periods.
            """
        }
        return html.Div([
            html.P(descriptions.get(hypothesis, ""), className="mb-0")
        ])

    def create_supporting_metrics_panel(self, df: pd.DataFrame, 
                                     period: str, hypothesis: str) -> html.Div:
        """Create enhanced supporting metrics visualization"""
        metrics = {}
        
        # Add bank type distribution with percentage
        total_banks = len(df['cert'].unique())
        bank_type_counts = df['bank_type'].value_counts()
        metrics['Bank Type Distribution'] = {
            f"{k} ({v/total_banks*100:.1f}%)": v 
            for k, v in bank_type_counts.items()
        }
        
        # Add size distribution
        asset_bins = [0, 1e9, 10e9, 100e9, float('inf')]  # 1B, 10B, 100B+
        asset_labels = ['Small (<$1B)', 'Medium ($1-10B)', 'Large ($10-100B)', 'Very Large (>$100B)']
        df['asset_size_class'] = pd.cut(df['asset'], 
                                      bins=asset_bins,
                                      labels=asset_labels)
        size_counts = df['asset_size_class'].value_counts()
        metrics['Asset Size Distribution'] = {
            k: v for k, v in size_counts.items()
        }
        
        # Hypothesis-specific metrics
        if hypothesis == 'h1':
            metrics['CRE Analysis'] = {
                'Median CRE Concentration': f"{df['cre_concentration'].median():.1f}%",
                'Mean ECL Change': f"{df[f'ecl_coverage_change_{period}'].mean():.1f}%",
                'CRE-ECL Correlation': f"{df['cre_concentration'].corr(df[f'ecl_coverage_change_{period}']):.3f}",
                'High CRE Count': f"{len(df[df['cre_concentration'] > df['cre_concentration'].median()])} banks"
            }
        elif hypothesis == 'h2':
            consumer_median = df['consumer_loan_ratio'].median()
            metrics['Consumer Loan Analysis'] = {
                'Median Consumer Ratio': f"{consumer_median:.1f}%",
                'High Consumer Banks': f"{len(df[df['consumer_loan_ratio'] > consumer_median])} banks",
                'Low Consumer Banks': f"{len(df[df['consumer_loan_ratio'] <= consumer_median])} banks",
                'Consumer-ECL Correlation': f"{df['consumer_loan_ratio'].corr(df[f'ecl_coverage_change_{period}']):.3f}"
            }
        elif hypothesis == 'h3':
            tier1_median = df['tier1_ratio'].median()
            metrics['Capital Analysis'] = {
                'Median Tier 1 Ratio': f"{tier1_median:.1f}%",
                'Mean ECL Change': f"{df[f'ecl_coverage_change_{period}'].mean():.1f}%",
                'Capital-ECL Correlation': f"{df['tier1_ratio'].corr(df[f'ecl_coverage_change_{period}']):.3f}",
                'Well-Capitalized Banks': f"{len(df[df['tier1_ratio'] > tier1_median])} banks"
            }
        elif hypothesis == 'h4':
            metrics['Model Performance'] = {
                'Mean Prediction Error': f"{df['prediction_error'].mean():.1f}%",
                'Median Prediction Error': f"{df['prediction_error'].median():.1f}%",
                'Error Std Dev': f"{df['prediction_error'].std():.1f}%",
                'Within 10% Accuracy': f"{(abs(df['prediction_error']) <= 10).mean()*100:.1f}%"
            }
            
        # Create formatted output with enhanced styling
        content = []
        for section_name, section_metrics in metrics.items():
            content.extend([
                html.H6(section_name, className="text-primary"),
                html.Hr(className="mt-2 mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6(k, className="card-subtitle mb-2 text-muted"),
                                html.P(str(v), className="card-text h5")
                            ])
                        ], className="mb-3")
                    ], width=6)
                    for k, v in section_metrics.items()
                ], className="g-2")
            ])
            
        return html.Div(content)

    def create_stats_panel(self, results: Dict) -> html.Div:
        """Create enhanced statistical results panel"""
        if not results:
            return html.Div("No results available.")
        
        if 'error' in results:
            return html.Div([
                html.H6("Error in Analysis", className="text-danger"),
                html.P(results['error'])
            ])
        
        content = [html.H6("Statistical Analysis Results", className="text-primary")]
        
        # Format results with enhanced styling
        for key, value in results.items():
            if isinstance(value, dict):
                content.append(html.H6(key, className="mt-3"))
                content.append(html.Hr(className="mt-2 mb-3"))
                
                # Create cards for nested metrics
                content.append(dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6(sub_key, className="card-subtitle mb-2 text-muted"),
                                html.P(
                                    f"{sub_value:.3f}" if isinstance(sub_value, float)
                                    else str(sub_value),
                                    className="card-text h5"
                                )
                            ])
                        ], className="mb-3")
                    ], width=6)
                    for sub_key, sub_value in value.items()
                ], className="g-2"))
            else:
                content.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.H6(key, className="card-subtitle mb-2 text-muted"),
                            html.P(
                                f"{value:.3f}" if isinstance(value, float)
                                else str(value),
                                className="card-text h5"
                            )
                        ])
                    ], className="mb-3")
                )
        
        return html.Div(content)

    def prepare_download_data(self, df: pd.DataFrame, stress_period: str, hypothesis: str) -> pd.DataFrame:
        """Prepare comprehensive DataFrame for download"""
        # Basic columns always included
        columns = ['cert', 'date', 'bank_type', 'bank', 'abbreviated_name', 'asset']
        
        # Add hypothesis-specific columns
        if hypothesis == 'h1':
            columns.extend(['cre_concentration', f'ecl_coverage_change_{stress_period}'])
        elif hypothesis == 'h2':
            columns.extend(['consumer_loan_ratio', f'ecl_coverage_change_{stress_period}'])
        elif hypothesis == 'h3':
            columns.extend(['tier1_ratio', f'ecl_coverage_change_{stress_period}'])
        elif hypothesis == 'h4':
            columns.extend([
                'lnatres', 'predicted_ecl', 'prediction_error',
                'model_r2_train', 'model_r2_test', 'top_predictors'
            ])
        
        # Create download DataFrame
        download_df = df[columns].copy()
        
        # Add bank type and size classification
        download_df['asset_size_class'] = pd.qcut(
            download_df['asset'], 
            q=4, 
            labels=['Small', 'Medium', 'Large', 'Very Large']
        )
        
        # Format date column
        download_df['date'] = download_df['date'].dt.strftime('%Y-%m-%d')
        
        # Sort by bank and date
        download_df = download_df.sort_values(['bank', 'date'])
        
        return download_df

# Initialize warnings and data processing
warnings.filterwarnings('ignore', message='.*Unverified HTTPS.*')

def main():
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
        return app
    else:
        print("Error: Unable to initialize dashboard due to empty dataset.")
        return None

if __name__ == "__main__":
    app = main()
    if app:
        app.run_server(debug=False)
