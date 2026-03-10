import xgboost as xgb
import pandas as pd
from src.models.data_models import ShipmentModel
from src.agents.agents.base_agent import BaseAgent
import numpy as np
import mlflow.pyfunc
import os
import platform

class PricingAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="PricingAgent")
        model_path=os.path.join("trained_models", "pricing_xgb_model.json")

        self.model=xgb.Booster()
        if os.path.exists(model_path):
            self.model.load_model(model_path)
            print(f"XGBoost model loaded successfully from: {model_path}")
        else:
            print(f"ERROR: Model file not found at: {model_path}")
            # List files to help you debug in the Docker logs
            print(f"Current directory contents: {os.listdir('.')}")
        self.feature_cols=['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'total_weight_kg', 'distance_km', 'hour', 'day_of_week', 'is_holiday', 'duration_min', 'traffic_density_score', 'is_rush_hour', 'is_weekend', 'is_high_demand', 'type_bicycle', 'type_e_scooter', 'type_truck', 'type_van']
    
    def process(self, shipment: ShipmentModel) -> ShipmentModel:
        data_dict=shipment.model_dump(by_alias=True)
        df=pd.DataFrame([data_dict])
       
        X = df[self.feature_cols]
        print(f"DEBUG: Feature Vector -> {X.iloc[0].to_dict()}")
        dmat=xgb.DMatrix(X)

        #xgbosst predicts abse price
        base_prediction=self.model.predict(dmat)[0]
        actual_price = np.expm1(base_prediction)
        shipment.predicted_base_price=round(float(actual_price),2)
        shipment.raw_model_prediction= shipment.predicted_base_price
        surge_multiplier= self._calculate_market_surge(shipment)
  
        shipment.final_market_price=round(shipment.predicted_base_price*surge_multiplier,2)
        print(f"final_pricing_agent_price:{shipment.final_market_price}")
        shipment.weather_factor = surge_multiplier

        trace_entry=(
             f"[{self.name} Success] ->"
             f"base_price:{shipment.predicted_base_price},"
             f"final_price:{shipment.final_market_price},"
             f"weather_condition:{shipment.weather_condition},"
             f"weather_factor:{shipment.weather_factor}"
        )
        
        shipment.agent_trace.append(f"\n{trace_entry}\n")
       
        return shipment

    def _calculate_market_surge(self, shipment: ShipmentModel) -> float:
        """Determines the market multiplier based on environmental factors"""
        multiplier = 1.0
        
        # Weather Surge
    
        if shipment.weather_condition == "Rain":
            multiplier += 0.2  # 10% Surge
        elif shipment.weather_condition == "Snow":
            multiplier += 0.4  # 30% Surge
        elif shipment.weather_condition == "Storm":
            multiplier += 0.6  # 50% Surge
        elif shipment.weather_condition == "Fog":
            multiplier += 0.1  # 50% Surge
        elif shipment.weather_condition == "Overcast":
            multiplier += 0.05 # 50% Surge
        else:
            multiplier=1.0
            
        return multiplier
