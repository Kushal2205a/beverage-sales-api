import json
import joblib 
import pandas as pd 
import numpy as np

import holidays 

us_holidays = holidays.US()

def recursive_xgb_forecast(model, state_df,  feature_columns, state,horizon = 8):
    df = state_df.copy().sort_values('Date').reset_index(drop=True)
    
    for _ in range(horizon):
        last_row = df.iloc[-1]
        next_date = last_row['Date']+ pd.Timedelta(weeks = 1 )
        
        def safe_lag(series, lag):
            return series.iloc[-lag] if len(series) >= lag else series.iloc[0]
        
        recent = df['Total_log'].iloc[-4:]
        
        new_row = {
            
            'Date' : next_date,
            'Total_log' : None,
            'lag_1' : safe_lag(df['Total_log'], 1),
            'lag_7' : safe_lag(df['Total_log'], 7),
            'lag_30' : safe_lag(df['Total_log'], 30),
            
            'rolling_mean_4' :recent.mean(),
            'rolling_std_4': recent.std() if len(recent) > 1 else 0,
            
            'month' : next_date.month,
            'week_of_year':next_date.isocalendar().week,
            
            'is_holiday_week':0
        }
        
        for col in feature_columns:
            if col.startswith('State_'):
                new_row[col] = 1 if col == f'State_{state}' else 0 
        
        x_new = pd.DataFrame([new_row])
        x_new = x_new.reindex(columns = feature_columns, fill_value = 0 )
        
        pred_log = model.predict(x_new)[0]
        new_row['Total_log']  = pred_log
        
        df = pd.concat([df,pd.DataFrame([new_row])], ignore_index= True)
    return np.expm1(df['Total_log'].iloc[-horizon:].values)
         

def generate_all_forecasts(clean_df, xgb_model, prophet_models, arima_models,feature_columns,horizon = 8):
    with open('tournament_results_v2.json','r') as f:
        tournament = json.load(f)
        
    all_forecasts = {}
    
    for state, results in tournament.items():
        best = results['Current_Best']
        state_df = clean_df[clean_df['State'] == state].sort_values('Date')
        
        last_date = state_df['Date'].max()
        
        future_dates = pd.date_range(start = last_date, periods = horizon+1 , freq= 'W-SUN')[1:]
        
        if best == 'Prophet':
            model = prophet_models[state]
            future = model.make_future_dataframe(periods = horizon, freq='W-SUN')
            
            def is_holiday_week(date):
                for i in range(7):
                    check_date = date - pd.Timedelta(days = i )
                    if check_date in us_holidays:
                        return 1 
                return 0 
            
            future['is_holiday_week'] = future['ds'].apply(is_holiday_week)
            forecast = model.predict(future)

            
            preds = forecast['yhat'].tail(horizon).values
            
        elif best == 'XGBoost':
            preds = recursive_xgb_forecast(xgb_model,state_df, feature_columns, state, horizon)
            
        elif best == 'ARIMA':
            model = arima_models[state]
            preds_log = model.predict(n_periods = horizon)
            preds= np.expm1(preds_log)
            
        all_forecasts[state] = [{"date" : str(d.date()) , "predicted_sales" : round(float(p),2),"predicted_sales_millions" : round(float(p)/1e6,2) } for d,p in zip(future_dates,preds)]
        
        print(f"{state} ({best}) : {[round(float(p)/1e6,1) for p in preds] } M")
        
    with open('forecasts.json','w') as f:
        json.dump(all_forecasts,f, indent = 4 )
        
    print("\n Saved")
    return all_forecasts
        
