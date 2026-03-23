from src.agents.agents.base_agent import BaseAgent
from src.models.data_models import ShipmentModel
import json
import math
import os
from src.utils.prediction_data_validator import DataValidationError

class CriticAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="CriticAgent")
        self.mapping_path = os.path.join("data", "transformed","traffic_mapping.json")
            # Match the exact path where your NYCFeatureEngineer saved the JSON
        try:
            with open(self.mapping_path ,"r") as f:
                    self.traffic_memory = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Traffic mapping not found for {self.name}. Using defaults.")
            self.traffic_memory = {}

        self.v_base = {'e_scooter': 2.0, 'bicycle': 3.0, 'van': 15.0, 'truck': 45.0}
        self.v_rate = {'e_scooter': 0.5, 'bicycle': 1.0, 'van': 2.5, 'truck': 5.0}

    def process(self, shipment: ShipmentModel) -> ShipmentModel:
        # --- CONDITION 1: DOCUMENT EXTRACTION ---
        overrides = [] 
        attempt=0
        if shipment.total_weight_kg <= 0 or shipment.parcel_count <= 0:
            shipment.is_verified = False  
            shipment.extraction_attempts+=1
            reasons = []
            if shipment.total_weight_kg <= 0:
                reasons.append("Weight is 0.0 or missing. Look for keywords like 'lbs', 'kg', 'heavy', or 'pounds'.")
            if shipment.parcel_count <= 0:
                reasons.append("Parcel count is 0. Search for 'items', 'boxes', 'units', or 'pallets'.")
            
            # Construct the instructional prompt for the LLM
            feedback_note = (
                f"CRITIC_REJECT: Logical validation failed. "
                f"Issues detected: {'; '.join(reasons)} "
                f"Current Extraction -> Weight: {shipment.total_weight_kg}kg, Items: {shipment.parcel_count}. "
                f"Instruction: Re-analyze the raw text for these specific fields. If truly missing, return 'NULL'."
            )

            if feedback_note:
                if shipment.extraction_attempts > 3:
                    print("Document processor failed.Manual entry required")
                    error_msg = (
                    "CRITICAL: Document processor failed after 3 attempts. "
                    "Manual entry required for Weight/Parcel count. "
                    f"Last Extraction State -> Weight: {shipment.total_weight_kg}, Items: {shipment.parcel_count}"
                    )
                    print(error_msg) # Console log for Docker/Backend
                    raise DataValidationError(error_msg)
            shipment.agent_trace.append(f"CRITIC_FEEDBACK: {feedback_note}")
            return shipment # Orchestrator will now see 'is_verified=False' and retry

        # --- CONDITION 2: ROUTE & ETA OVERRIDE ---
        correct_v = self._get_correct_vehicle(shipment.distance_km, shipment.total_weight_kg)
        if shipment.vehicle_type != correct_v:
            overrides.append(f"Vehicle: {shipment.vehicle_type} -> {correct_v}")
            shipment.vehicle_type = correct_v
            self._sync_vehicle_flags(shipment)
        

        key = f"{shipment.hour}_{shipment.day_of_week}_{shipment.is_holiday}"
        stats = self.traffic_memory.get(key, {"actual_speed_kmh": 20.0})
        theoretical_duration = (shipment.distance_km / stats['actual_speed_kmh']) * 60

        if shipment.duration_min <= 0 or abs(shipment.duration_min - theoretical_duration) > (theoretical_duration * 0.8):
            overrides.append(f"Duration: {shipment.duration_min}min -> {theoretical_duration}min")
            shipment.duration_min = round(theoretical_duration, 2)

        # --- CONDITION 3: PRICING SANITY ---
        theory_price = self._calculate_theoretical_price(shipment)
        diff = abs(shipment.raw_model_prediction - theory_price) / theory_price
        
        if diff > 0.10:
            overrides.append(f"Price: ${shipment.predicted_base_price} -> ${theory_price} (Rule-based correction)")
            shipment.predicted_base_price = theory_price
            shipment.final_market_price = round(theory_price * shipment.weather_factor, 2)
        
        print("critic_agent verification completed")
        shipment.is_verified = True
        if overrides:
            # Log exactly what changed
            shipment.agent_trace.append(f"{self.name}: {'; '.join(overrides)}")
        else:
            shipment.agent_trace.append("CriticAgent_Success: No overrides required.")
        return shipment

    def _get_correct_vehicle(self, d, w):
        if w > 150:
            return 'truck'
        elif w > 20:
            return 'van'
        
        # Light weights (w <= 20)
        elif w <= 20.0:
            if d <= 3.0:
                return 'e_scooter'
            elif 3.0 < d <= 10.0:  # Matches the new 10km Bicycle rule
                return 'bicycle'
            else:
                return 'van'       # Default for long distances > 10km
            
        return 'van' # Ultimate fallback

    def _sync_vehicle_flags(self, s: ShipmentModel):
        s.type_e_scooter, s.type_bicycle, s.type_van, s.type_truck = 0, 0, 0, 0
        setattr(s, f"type_{s.vehicle_type}", 1)

    def _calculate_theoretical_price(self, s: ShipmentModel):
        base = self.v_base.get(s.vehicle_type, 45.0)
        km_r = self.v_rate.get(s.vehicle_type, 5.0)
        surge = 1.0 + (s.is_rush_hour * 0.2) + (s.is_weekend * 0.15) + (s.is_holiday * 0.4)
        congestion = 1.25 if s.traffic_density_score < 0.5 else 1.0
        weight_fee = (s.total_weight_kg - 5) * 0.5 if s.total_weight_kg > 5 else 0
        
        return round((base + (s.distance_km * km_r) + (s.parcel_count * 1.5) + (s.duration_min * 0.2) + weight_fee) * surge * congestion, 2)

    def haversine(self,lat1, lon1, lat2, lon2):
        # Fallback if OSRM is down
        R = 6371 # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c