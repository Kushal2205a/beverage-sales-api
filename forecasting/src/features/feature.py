import pandas as pd
import numpy as np
import holidays
import warnings
warnings.filterwarnings('ignore')

def clean_sales_data(df):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values(['State', 'Date'])
    
    us_holidays = holidays.US()
    processed_states = []
    
    for state in df['State'].unique():
        state_df = df[df['State'] == state].copy()
        
    
        numeric_cols = state_df[['Date', 'Total']].set_index('Date')
        state_df_resampled = numeric_cols.resample('W-SUN').sum(min_count=1)
        state_df_resampled['Total'] = state_df_resampled['Total'].interpolate(method='linear')
        
    
        state_df_resampled['Total_log'] = np.log1p(state_df_resampled['Total'])
        
        state_df_resampled['State'] = state
        state_df_resampled['Category'] = 'Beverages'
        
        
        state_df_resampled['lag_1'] = state_df_resampled['Total_log'].shift(1)
        state_df_resampled['lag_7'] = state_df_resampled['Total_log'].shift(7)
        state_df_resampled['lag_30'] = state_df_resampled['Total_log'].shift(30)
        
        
        state_df_resampled['rolling_mean_4'] = state_df_resampled['Total_log'].shift(1).rolling(window=4).mean()
        state_df_resampled['rolling_std_4'] = state_df_resampled['Total_log'].shift(1).rolling(window=4).std()
        
        state_df_resampled = state_df_resampled.reset_index()
        
        
        state_df_resampled['is_holiday_week'] = state_df_resampled['Date'].apply(
            lambda d: int(any((d - pd.Timedelta(days=i)) in us_holidays for i in range(7)))
        )
        state_df_resampled['month'] = state_df_resampled['Date'].dt.month
        state_df_resampled['week_of_year'] = state_df_resampled['Date'].dt.isocalendar().week.astype(int)
        
        processed_states.append(state_df_resampled)
        
    final_df = pd.concat(processed_states).dropna().reset_index(drop=True)
    return final_df