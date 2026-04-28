from pmdarima import auto_arima
import pandas as pd 
import numpy as np 
import json 
import joblib 
import warnings 
warnings.filterwarnings('ignore')

def calculate_smape(actual,pred):
    denom = np.abs(actual) + np.abs(pred)
    denom = np.where(denom == 0,1,denom)
    return 100 * np.mean(2* np.abs(pred-actual)/denom)

def train_local_arima(clean_df, forecast_horizon = 8, tournament_path = 'tournament_results_v2.json'):
    try:
        with open(tournament_path, 'r') as f:
            tournament = json.load(f)   
    except FileNotFoundError:
        tournament = {}
        
    arima_models = {}
    
    for state in clean_df['State'].unique():
        state_df = clean_df[clean_df['State'] == state].copy().sort_values('Date')
        
        cutoff = state_df['Date'].max()-pd.Timedelta(weeks = forecast_horizon)
        train = state_df[state_df['Date'] <= cutoff]
        test = state_df[state_df['Date'] > cutoff]
        
        y_train = train['Total'].values
        
        try:
            model = auto_arima(
                y_train,
                exogenous = train[['is_holiday_week']],
                seasonal = True,
                m = 52,
                max_p=2,max_q=2,max_d=1,max_P=1,max_Q=1,max_D=1,
                suppress_warnings = True,
                error_action = 'ignore',
                stepwise = True )
            preds = model.predict(n_periods= forecast_horizon,exogenous = test[['is_holiday_week']])
            pred_data = np.clip(preds,0,None)
            actual_dat = test['Total'].values
            
            min_len = min(len(actual_dat),len(pred_data))
            actual_dat= actual_dat[:min_len]
            pred_data= pred_data[:min_len]
            
            smape = calculate_smape(actual_dat,pred_data)
            arima_models[state] = model
            
            if state not in tournament: 
                tournament[state] = {}
            tournament[state]['ARIMA_SMAPE'] = round(smape,2)
            
            
            
            score = {k:v for k,v in tournament[state].items() if k.endswith('_SMAPE') and v is not None}
            if score:
                best_model_key = min(score, key=score.get)
                tournament[state]['Current_Best'] = best_model_key.replace('_SMAPE','') 
            print(f"{state} : ARIMA SMAPE score {round(smape,2)} and the Winner : {tournament[state]['Current_Best']}")
            
        except Exception as e:
            print(f"{state} : Arima failed, {e}")
            if state not in tournament:
                tournament[state] = {}
                tournament[state]['ARIMA_SMAPE'] = None 
    with open('final_model_routing.json', 'w') as f:
        json.dump(tournament,f,indent= 4)
    joblib.dump(arima_models,"arima_models.pkl")
    
    print("Training done")
    
    return arima_models, tournament  
            