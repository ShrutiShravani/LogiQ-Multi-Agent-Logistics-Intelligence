import json
import holidays
from datetime import datetime
from src.models.data_models import ShipmentModel
import os

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.us_holidays = holidays.US(state='NY')
        try:
            self.mapping_path = os.path.join("data", "transformed","traffic_mapping.json")
            # Match the exact path where your NYCFeatureEngineer saved the JSON
            with open(self.mapping_path ,"r") as f:
                self.traffic_memory = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Traffic mapping not found for {self.name}. Using defaults.")
            self.traffic_memory = {}
    
    def enrich_metadata(self, shipment: ShipmentModel):
        """
        Senior-Level Feature Engineering.
        Syncs Live Shipment with Training Logic.
        """
        dt = shipment.pickup_time
        shipment.hour = dt.hour
        shipment.day_of_week = dt.weekday()
        
        # 1. Holiday Logic (Must match training!)
        shipment.is_holiday = 1 if dt.date() in self.us_holidays else 0
        
        # 2. Rush Hour (Physical Clock Logic)
        # 8-10 AM or 4-7 PM
        shipment.is_rush_hour = 1 if (8 <= shipment.hour <= 10 or 16 <= shipment.hour <= 19) else 0
        shipment.is_weekend = 1 if shipment.day_of_week >= 5 else 0

        # 3. High Demand (Economic Logic)
        # Rule: Weekday Rush OR Weekend OR Holiday
        shipment.is_high_demand = 1 if (
            (shipment.is_rush_hour == 1 and shipment.is_weekend == 0) or 
            (shipment.is_weekend == 1) or 
            (shipment.is_holiday == 1)
        ) else 0

        #if self.name not in shipment.agent_trace:
            #shipment.agent_trace.append(self.name)
        
        return shipment
    
    def get_vehicle_type(self, shipment: ShipmentModel) -> ShipmentModel:
        """
        Logic: Selects vehicle based on the same weight/distance thresholds as training.
        """
        # Reset OHE fields to 0
        shipment.type_truck = shipment.type_van = shipment.type_bicycle = shipment.type_e_scooter = 0
        
        w = shipment.total_weight_kg
        d = shipment.distance_km  # Ensure Route Agent has filled this!

        # EXACT MATCH TO TRAINING CONDITIONS:
        if w > 150:
            shipment.vehicle_type, shipment.type_truck = "truck", 1
        elif w > 20:
            shipment.vehicle_type, shipment.type_van = "van", 1

        # 2. LIGHT WEIGHTS (0 - 20kg) - Choose by Distance
        elif w <= 20.0:
            if d <= 3.0:
                shipment.vehicle_type, shipment.type_e_scooter = "e_scooter", 1
            elif 3.0 < d <= 10.0: # This perfectly captures your 7.5km / 2.5kg case
                shipment.vehicle_type, shipment.type_bicycle = "bicycle", 1
            else:
                # Distance is > 10km, too far for a bike
                shipment.vehicle_type, shipment.type_van = "van", 1

        # 3. FINAL SAFETY FALLBACK
        else:
            shipment.vehicle_type, shipment.type_van = "van", 1

        return shipment

    def get_historical_traffic(self, shipment: ShipmentModel) -> float:
        """
        Look up traffic density from training memory using the 3-key format: hour_day_holiday
        """
        # Create key matching your JSON: "hour_day_holiday"
        key = f"{shipment.hour}_{shipment.day_of_week}_{shipment.is_holiday}"
        
        # Look up stats
        stats = self.traffic_memory.get(key, {})
        
        # Get score, default to 1.0 (Normal) if not found
        score = stats.get("traffic_density_score",1.0)
        
        shipment.traffic_density_score = round(float(score), 2)
        print(f"traffic_Density_score:{shipment.traffic_density_score}")
        return shipment.traffic_density_score
    
    def log_to_tracing(self,shipment:ShipmentModel):
        """
        Placeholder for mflow/langsmith tracking
        """
        print(f"[{self.name}] Processing shipment:{shipment.shipment_id}")
        