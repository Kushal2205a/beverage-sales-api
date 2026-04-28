import pandas as pd
import numpy as np
import xgboost as xgb
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def calculate_smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1, denom)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)

def train_global_xgboost(clean_df, local_results_path='local_model_results.json', forecast_horizon=8):
    try:
        with open(local_results_path, 'r') as f:
            tournament = json.load(f)
    except FileNotFoundError:
        print("Running without local baselines.")
        tournament = {}

    
    
    ml_df = clean_df.copy().sort_values(['State', 'Date'])
    ml_df['State_String'] = ml_df['State'] 
    
    cutoff_date = ml_df['Date'].max() - pd.Timedelta(weeks=forecast_horizon)
    train = ml_df[ml_df['Date'] <= cutoff_date].copy()
    test = ml_df[ml_df['Date'] > cutoff_date].copy()
    
    train = pd.get_dummies(train, columns=['State'], drop_first=False)
    test = pd.get_dummies(test, columns=['State'], drop_first=False)
    
    
    missing_cols = set(train.columns) - set(test.columns)
    for col in missing_cols:
        test[col] = 0
    test = test[train.columns] 
    
    cols_to_drop = ['Date', 'Total', 'Total_log', 'Category', 'State_String']
    X_train = train.drop(columns=cols_to_drop)
    y_train = train['Total_log'] 
    X_test = test.drop(columns=cols_to_drop)
    
    
    assert list(X_train.columns) == list(X_test.columns), "Train/Test feature mismatch!"
    
    print(f"Training xgboost on {len(X_train)} rows")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6, 
        random_state=42, objective='reg:squarederror'
    )
    xgb_model.fit(X_train, y_train)
    

    print("\nTop 10 Important features ")
    importance = xgb_model.feature_importances_
    feat_imp = sorted(zip(X_train.columns, importance), key=lambda x: x[1], reverse=True)
    for f, imp in feat_imp[:10]:
        print(f"{f}: {round(imp, 4)}")
        
    print("\n Behavioral feature")
    for f, imp in feat_imp:
        if 'State_' not in f:
            print(f"{f}: {round(imp, 4)}")
            break
    print("\n\n")
    
    real_scale_predictions = np.clip(np.expm1(xgb_model.predict(X_test)), 0, None)
    
    test_results = test[['Date', 'State_String', 'Total']].copy()
    test_results['XGB_Pred'] = real_scale_predictions
    test_results.rename(columns={'Total': 'Actual_Total'}, inplace=True)
    
    for state in test_results['State_String'].unique():
        state_data = test_results[test_results['State_String'] == state]
        xgb_smape = calculate_smape(state_data['Actual_Total'].values, state_data['XGB_Pred'].values)
        
        if state not in tournament: tournament[state] = {}
        tournament[state]['XGBoost_SMAPE'] = round(xgb_smape, 2)
        
        local_scores = [tournament[state].get('Prophet_SMAPE'), tournament[state].get('ARIMA_SMAPE')]
        local_scores = [s for s in local_scores if s is not None]
        best_local = min(local_scores) if local_scores else float('inf')
        
        tournament[state]['Current_Best'] = 'XGBoost' if xgb_smape < best_local else tournament[state].get('Local_Winner', 'Unknown')

    with open('tournament_results_v2.json', 'w') as f:
        json.dump(tournament, f, indent=4)
        
    print("Training done")
    xgb_model.save_model("global_xgboost.json")
    return xgb_model, tournament

