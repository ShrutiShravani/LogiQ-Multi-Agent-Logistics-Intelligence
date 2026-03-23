from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional,List,Dict,Any

class ShipmentModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # 1. FROM WAYBILL (Document Agent)
    shipment_id: str
    origin_address: str
    destination_address: str
    total_weight_kg:float = 0.0
    item_category: str = "standard_parcels"
    parcel_count: int = Field(..., alias="passenger_count")
    pickup_time: datetime
    type_truck: int = 0
    type_van: int = 0
    type_bicycle: int = 0
    type_e_scooter: int = 0
    vehicle_type:str="bicycle"
    route_options:List[Dict[str,Any]]=[]
    selected_route_index: int = 0
    delay_delta: float = 0.0
    extraction_attempts: int = 0


    # 2. FROM ROUTE AGENT & UTILS
    # These map directly to your XGBoost training columns
    pickup_latitude: float = 0.0
    pickup_longitude: float = 0.0
    dropoff_latitude: float = 0.0
    dropoff_longitude: float = 0.0
    distance_km: float = 0.0
    duration_min: float = 0.0
    
    # 3. FROM FEATURE ENGINEERING (Applied on-the-fly)
    hour: int = 0
    day_of_week: int = 0
    is_holiday: int = 0  # <--- ADD THIS to match your CSV
    is_rush_hour: int = 0
    is_weekend: int = 0
    is_high_demand: int = 0
    traffic_density_score: float = 1.0 # Match training default

    # 4. DEMAND FORECASTING INPUTS
    weather_condition: str="sunny"
    weather_factor: float = 1.0 # Default: 1.0 (Off-peak)
      
    # 5. FINAL OUTPUTS
    raw_model_prediction: float =0.0
    predicted_base_price: float = 0.0 # Raw XGBoost output
    final_market_price: float = 0.0   # XGBoost * Demand Surge

    is_verified: bool = False
    agent_trace: List[str] = []