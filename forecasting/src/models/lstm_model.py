import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Input 
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler 
import pandas as pd 
import numpy as np 
import json 
import warnings 
import os 

warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

def calculate_smape(actual,pred):
    denom = np.abs(actual) + np.abs(pred)
    denom = np.where(denom == 0,1,denom)
    
    return 100 * np.mean(2 * np.abs(pred - actual)/denom)

def create_sequences(state_df,feature_cols,seq_length = 8):
    x,y,dates,states,actual_total= [],[],[],[],[]
    
    features = state_df[feature_cols].values 
    targets = state_df['Total_log'].values
    
    date_vals= state_df['Date'].values 
    state_name = state_df['State'].iloc[0]
    totals = state_df['Total'].values
    
    
    for i in range(len(state_df) - seq_length):
        x.append(features[i:i+seq_length])
        y.append(targets[i+seq_length])
        
        dates.append(date_vals[i+seq_length])
        states.append(state_name)
        
        actual_total.append(totals[i+seq_length])
        
    return np.array(x),np.array(y),np.array(dates),np.array(actual_total),np.array(states)

def train_global_lstm(clean_df, results_path = 'final_model_routing.json',forecast_horizon = 8, seq_length = 8):
    try:
        with open(results_path,'r') as f:
            tournament = json.load(f)
    except FileNotFoundError:
        print("File not found")
        tournament= {}
        

    tournament = {
    k: v for k, v in tournament.items()
    if isinstance(k, str) and not k.replace('.', '', 1).isdigit()
    }
        
    clean_df = clean_df.sort_values(['State','Date']).copy()
    
    cutoff_date = clean_df['Date'].max() - pd.Timedelta(weeks = forecast_horizon)
    
    feature_cols = [c for c in clean_df.columns if c not in ['Date','Total','Total_log','State','Category']]
    
    train = clean_df['Date'] <= cutoff_date
    scaler = StandardScaler()
    scaler.fit(clean_df.loc[train, feature_cols])
    
    scaled_df = clean_df.copy()
    scaled_df[feature_cols] = scaler.transform(clean_df[feature_cols])
    
    all_x, all_y, all_dates, all_states,all_actual_totals = [],[],[],[],[]
    
    for state in scaled_df['State'].unique():
        state_data = scaled_df[scaled_df['State'] == state]
        
        if len(state_data) <= seq_length:
            continue 
        
        x,y,d,t,s = create_sequences(state_data, feature_cols, seq_length)
        
        all_x.append(x)
        all_y.append(y)
        
        all_dates.append(d)
        all_states.append(s)
        
        all_actual_totals.append(t)
        
    x_global = np.concatenate(all_x)
    y_global = np.concatenate(all_y)
    
    dates_global = np.concatenate(all_dates)
    states_global = np.concatenate(all_states)
    
    actual_total_global = np.concatenate(all_actual_totals)
    
    train_idx = dates_global <= np.datetime64(cutoff_date)
    test_idx = dates_global > np.datetime64(cutoff_date)
    
    x_train, y_train = x_global[train_idx],y_global[train_idx]
    x_test,y_test = x_global[test_idx],y_global[test_idx]
    
    test_states = states_global[test_idx]
    test_actual_totals = actual_total_global[test_idx]
    
    model = Sequential([
        Input(shape = (seq_length,len(feature_cols))),
        LSTM(32),
        Dense(16, activation = 'relu'),
        Dense(1,activation = 'linear')
    ])
    
    model.compile(optimizer = 'adam', loss = 'mse')
    
    early_stop = EarlyStopping(monitor = 'val_loss', patience = 3 , restore_best_weights = True)
    
    model.fit(x_train,y_train,
              epochs = 20, 
              batch_size = 32, 
              validation_split = 0.1, 
              callbacks = [early_stop],
              verbose = 1 )
    
    log_preds = model.predict(x_test).flatten()
    real_preds = np.clip(np.expm1(log_preds),0,None)
    
    print(test_states[:10])
    
    results = pd.DataFrame({
        'State' : test_states,
        'Actual': test_actual_totals,
        'Pred' : real_preds
        
    }) 
    
    results['Actual'] = pd.to_numeric(results['Actual'],errors='coerce')
    results['Pred'] = pd.to_numeric(results['Pred'],errors='coerce')
    
    for state in results['State'].unique():
        state_data = results[results['State'] == state]
        smape = calculate_smape(state_data['Actual'].values, state_data['Pred'].values )
        
        if state not in tournament:
            tournament[state] = {}
            
        tournament[state]['LSTM_SMAPE'] = round(smape,2)
        
        scores = {k:v for k,v in tournament[state].items() if k.endswith('_SMAPE') and v is not None}
        if scores:
            best_model_key = min(scores, key = scores.get)
            tournament[state]['Current_Best'] = best_model_key.replace('_SMAPE', '')
            
        print(f"{state} , LSTM_SMAPE : {round(smape,2)} and Overall Winner: {tournament[state]['Current_Best']}")
        
    with open('final_model_routing.json','w') as f :
        json.dump(tournament,f,indent = 4 )
        
    
    os.makedirs('models', exist_ok=True)
    
    model.save('models/global_lstm.keras')
    print("Done Training")
    return model,tournament 
            
        
        
        
            
    
    
