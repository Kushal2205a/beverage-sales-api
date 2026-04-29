from fastapi import FastAPI, HTTPException 

import json 
from pathlib import Path 


app = FastAPI(title = "Beverage Sales Forecasting API")

BASE_DIR = Path(__file__).resolve().parent.parent.parent

with open(BASE_DIR/'notebook'/'forecasts.json','r') as f:
    forecasts = json.load(f)
with open(BASE_DIR/'notebook'/'tournament_results_v2.json','r') as f:
    tournament = json.load(f)

STATES = set(forecasts.keys())

@app.get("/")
def root():
    return{
        "message": "Beverage Sales Forecast API",
        "endpoints" : ["/forecast/{state}", "/models", "/states"]
    }
    
@app.get("/health", tags = ["System"])
def health():
    return {
        "status" : "ok", "states_loaded" : len(STATES)
    }

@app.get("/states",tags= ["Metadata"])
def get_states():
    return {"states":sorted(list(STATES))}

@app.get("/models",tags= ["MODELS"])
def get_models():
    return{
        state: {
            "best_model":data["Current_Best"],
            "smape_scores":{k:v for k,v in data.items() if k.endswith("_SMAPE")}
        } for state,data in tournament.items()
    }
    
@app.get("/forecast/{state}",tags= ["Forecast"])
def get_forecast(state):
    state_title = state.replace("-"," ").strip().title()
    
    if state_title not in STATES:
        raise HTTPException(
            status_code = 404,
            detail = f" State '{state}' not found, use /states to see valid option "
        )
    return {
        "state" : state_title,
        "model_used": tournament[state_title]["Current_Best"],
        "forecast_horizon_weeks":len(forecasts[state_title]),
        "units":{
            "predicted_sales" : "USD",
            "predicted_sales_millions" : "millions_USD"
            
        },
        "forecasts":forecasts[state_title]
    }
    
