from prophet import Prophet 
import pandas as pd 
import numpy as np 
import json 
import warnings 
warnings.filterwarnings('ignore')


def calculate_smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0,1,  denom)
    return 100 * np.mean(2* np.abs(y_pred - y_true)/denom)

def train_local_prophet(clean_df, forecast_horizon = 8,tournament_path= 'tournament_results_v2.json'):
    try:
        with open('tournament_path','r') as f:
            tournament = json.load(f)
            
    except FileNotFoundError:
        tournament = {}
        
    prophet_models = {}
    
    for state in clean_df['State'].unique():
        state_df = clean_df[clean_df['State'] == state].copy().sort_values('Date')
        
        cutoff = state_df['Date'].max()- pd.Timedelta(weeks = forecast_horizon)
        
        train = state_df[state_df['Date'] <= cutoff]['Date','Total_log'].rename(columns = {'Date' : 'ds' , 'Total_log':'y'})
        
        test = state_df[state_df['Date'] > cutoff]
        
        model = Prophet(yearly_seasonality = True, weekly_seasonality= False, daily_seasonality= False,seasonality_mode = 'multiplicative')
        
        
        model.add_country_holidays(country_name='US')
        
        model.fit(train)
        
        future = model.make_future_dataframe(periods= forecast_horizon, freq='W-SUN')
        
        forecast = model.predict(future)
        
        forecast = forecast.set_index('ds')
        test = test.set_index('Date')
        
        preds_log = forecast.loc(test.index, 'yhat').values 
        
        pred_data = np.clip(np.expm1(preds_log),0,None)
        actual_dat = test['Total'].values 
        
        smape = calculate_smape(actual_dat,pred_data)
        
        if state not in tournament:
            tournament[state] = {}
        tournament[state]['Prophet_SMAPE']= round(smape,2)
        
        
        all_scores = {'XGBoost' : tournament[state].get('XGBoost_SMAPE', float('inf')), 'Prophet': round(smape,2),'ARIMA':tournament[state].get('ARIMA_SMAPE', float('inf'))}
        
        tournament[state]['Current_Best']= min(all_scores, key = all_scores.get)
        prophet_models[state] = model
        
        print(f"{state} : Prophet smape = {round(smape,2)}")
        
    with open('tournament_results_v2.json','w') as f :
        json.dump(tournament,f,indent = 4 )
        
    print("Done training")
    return prophet_models,tournament
        