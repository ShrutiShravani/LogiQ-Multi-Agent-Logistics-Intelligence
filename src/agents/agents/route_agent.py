from src.models.data_models import ShipmentModel
from src.agents.agents.base_agent import BaseAgent
import requests
import os
from dotenv import load_dotenv
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeopyError
import math
import time
from src.utils.cache import Logisticscache
import functools

load_dotenv()

def safe_api_call(retries=3, delay=1.5):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        # If we hit a rate limit (429) or timeout
                        if "429" in str(e) or "timeout" in str(e).lower():
                            print(f"Rate limit/Timeout. Retry {attempt+1} in {delay}s...")
                            time.sleep(delay * (attempt + 1)) # Exponential backoff
                        else:
                            raise e
                return None
            return wrapper
        return decorator

class RouteAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="RouteAgent")
        #self.osrm_url = os.getenv("OSRM_URL")
        self.geolocator = Nominatim(user_agent=os.getenv("APP_USER_AGENT"))
        #self.weather_url= os.getenv("WEATHER_API_URL")
        self.mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN")
        self.weather_key = os.getenv("VISUAL_CROSSING_KEY")
        self.cache = Logisticscache()

    
    @safe_api_call(retries=3, delay=2)
    def _get_coords(self, address: str):
        """Geocoding: Address to NYC Coordinates"""
        clean_address= re.sub(r'\(.*?\)', '', address)
        segments= clean_address.split(',')
        clean_address2= ", ".join(segments[:4]).strip()
        
        cached_coords = self.cache.get_geo(clean_address2)
        if cached_coords:
            print("cache coords found")
            return cached_coords[0], cached_coords[1]

        #if not redis ,proceed to api call
        for attempt in range(3):
            try:
                time.sleep(1.1)
                    #chcek redis for aaddress
                location = self.geolocator.geocode(
                        clean_address2, 
                        viewbox=[(40.47, -74.25), (40.91, -73.70)], # NYC Bounds
                        bounded=True,
                        timeout=20
                    )
                if location:
                    self.cache.set_geo(clean_address2, [location.latitude, location.longitude])
                    return float(location.latitude), float(location.longitude)
        
                # Fallback for complex addresses (try without bounding box)
                location_fallback = self.geolocator.geocode(clean_address2)
                if location_fallback:
                    self.cache.set_geo(clean_address2, [location_fallback.latitude, location_fallback.longitude])
                    return float(location_fallback.latitude), float(location_fallback.longitude)
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {str(e)}")
                # If it's a rate limit or timeout, wait and retry
                time.sleep(1.5) 
                continue
        return None,None
    
    def haversine_distance(self,lat1, lon1, lat2, lon2):
        # Fallback if OSRM is down
        R = 6371 # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    @safe_api_call(retries=3, delay=2)
    def get_weather_impact(self, lat, lon,date_str):
        """Hybrid Signal: Live NYC Weather Impact on Traffic Density"""
        WEATHER_RULES = {
                "snow": (0.30, "Snow"),
                "rain": (0.15, "Rain"),
                "thunder": (0.45, "Storm"),
                "fog": (0.10, "Fog"),
                "cloudy": (0.05, "Overcast")
            }
        try:
            condition= self.cache.get_weather(lat,lon,date_str)
            
            if not condition:
                api_key = os.getenv("VISUAL_CROSSING_KEY")
                headers = {"User-Agent": "LogiQ_Agent_v1/1.0"}
                base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    
                # We request only 'current' to save data/latency
                url = f"{base_url}/{lat},{lon}/{date_str}?key={api_key}&unitGroup=metric&include=days&elements=datetime,temp,icon"
                response = requests.get(url,headers=headers,timeout=5)

                if response.status_code != 200:
                    return 0.0, "Clear"

                data = response.json()
                # The 'days' array contains the weather for the specific date requested
                if "days" in data and len(data["days"]) > 0:
                    day_data = data["days"][0]
                    condition = day_data.get("icon", "clear-day")
                    print(f"DEBUG: Weather Condition for {date_str} -> {condition}")
                    self.cache.set_weather(lat,lon,date_str,label=condition)
                else:
                    return 0.0, "Clear"
            
            else:
                print(f"CACHE HIT: Found weather '{condition}' for {date_str}")

            for key,(penalty,label) in WEATHER_RULES.items():
                if key in condition.lower():
                    return penalty,label
            return 0.0, "Clear"

        except Exception as e:
            print(f"Weather API Warning: {str(e)}")
            return 0.0, "Clear"

    @safe_api_call(retries=3,delay=2)
    def get_mapbox_route(self,url):
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            print(f"Mapbox Error: {response.status_code} - {response.text}")
            raise Exception(f"Mapbox_Error_{response.status_code}")
        return response.json()
            
    def process(self, shipment: ShipmentModel) -> ShipmentModel:
        # 1. Geocoding
        if shipment.pickup_latitude and shipment.pickup_longitude:
            lat, lon = shipment.pickup_latitude, shipment.pickup_longitude
            d_lat, d_lon = shipment.dropoff_latitude, shipment.dropoff_longitude
        else:

            lat, lon = self._get_coords(shipment.origin_address)
            d_lat, d_lon = self._get_coords(shipment.destination_address)

        if not lat or not d_lat:
            shipment.agent_trace.append("RouteAgent_Error: Geocoding failed for one or both addresses.")
            shipment.is_verified = False # Critic will handle the retry/DLQ
            return shipment
          
        shipment.pickup_latitude, shipment.pickup_longitude = lat, lon
        shipment.dropoff_latitude, shipment.dropoff_longitude = d_lat, d_lon

        # 2. Routing (OSRM)
        try:
            base_url = "https://api.mapbox.com/directions/v5/mapbox/driving"
    
            # URL structure: {lon},{lat};{d_lon},{d_lat}
            route_url = f"{base_url}/{lon},{lat};{d_lon},{d_lat}?access_token={self.mapbox_token}&alternatives=true&overview=false"
            
            print(f"DEBUG: Mapbox Request -> {route_url}")
            
            data = self.get_mapbox_route(route_url)
            
            if data.get("code") == "Ok":
                shipment.route_options=[]
                # Distance is in meters, Duration in seconds
                for idx,r in enumerate(data['routes']):
                    base_distance_km= round(r['distance']/1000,2)
                    base_duration_km= round(r['duration']/60,2)

                   
                    shipment.route_options.append({
                        "route_index": idx,
                        "base_distance_km": base_distance_km,
                        "base_duration_min": base_duration_km,
                    })
                print(f"shipment_route:{shipment.route_options}")
                shipment.distance_km =shipment.route_options['base_distance_km'][0]
                shipment.duration_min = shipment.route_options['base_duration_min'][0]
                print(f"SUCCESS: {shipment.distance_km}km, {shipment.duration_min}min")
            else:
                raise Exception(f"Mapbox_Invalid_Response: {data.get('code')}")
       
               
        except Exception as e:
            shipment.agent_trace.append(f"CIRCUIT_BREAKER: OSRM API unreachable. Using Haversine Fallback.")
            shipment.distance_km = self.haversine_distance(shipment.pickup_latitude,shipment.pickup_longitude, shipment.dropoff_latitude, shipment.dropoff_longitude)
            shipment.duration_min = round((shipment.distance_km / 20) * 60, 2)
        
        # 3. VEHICLE SELECTION (CRITICAL SYNC)
        # Now that we have distance_km AND total_weight_kg, we select the vehicle
        # this ensures it matches the NYCFeatureEngineer.transform() logic exactly.
        shipment = self.get_vehicle_type(shipment)
      
        # 4. METADATA ENRICHMENT (Time, Rush Hour, Holiday, High Demand)
        shipment = self.enrich_metadata(shipment)
        print(shipment.is_holiday)
        print(shipment.is_high_demand)
        print(shipment.is_rush_hour)
        print(shipment.is_weekend)

        # 5. HYBRID TRAFFIC INTELLIGENCE
        # Get historical mean for this [hour_day_holiday]
        hist_density = self.get_historical_traffic(shipment)
        # Get real-time weather penalty
        pickup_date = shipment.pickup_time.strftime('%Y-%m-%d')
        weather_penalty, weather_label = self.get_weather_impact(lat, lon,pickup_date)
        print(f"weather_penalty:{weather_penalty} , weather_label :{weather_label} ")

        # Final Density = Historical Baseline - Weather Friction
        # min 0.1 to avoid division by zero
        # Try:
        weather_multiplier = 1.0 - weather_penalty
      
        shipment.traffic_density_score = max(0.1, round(hist_density * weather_multiplier, 2))
        print(shipment.traffic_density_score)
        
        # Real-world Duration Adjustment: If density is 0.5, trip takes 2x longer
        shipment.weather_condition = weather_label

        for opt in shipment.route_options:
            opt['adjusted_duration_min'] = round(opt['base_duration_min'] / shipment.traffic_density_score, 2)
            opt['delay_delta']= round(opt['adjusted_duration_min']-opt['base_duration_min'],2)

        shipment.duration_min = shipment.route_options[0]["adjusted_duration_min"]

        trace_entry = (
        f"[{self.name} Success] ->"
        f"hour:{shipment.hour},"
        f"day_of_week:{shipment.day_of_week},"
        f"is_holiday:{shipment.is_holiday},"
        f"is_rush_hour:{shipment.is_rush_hour},"
        f"is_weekend:{shipment.is_weekend},"
        f"is_high_demand:{shipment.is_high_demand},"
        f"Pickup_latitude:{shipment.pickup_latitude},"
        f"Pickup_longitude:{shipment.pickup_longitude},"
        f"Destination_latitude:{shipment.dropoff_latitude},"
        f"Destination_longitude:{shipment.dropoff_longitude},"
        f"Dist: {shipment.distance_km}km, "
        f"Dur: {shipment.duration_min}min, "
        f"Vehicle: {shipment.vehicle_type}, "
        f"Weather: {shipment.weather_condition}, "
        f"TrafficScore: {shipment.traffic_density_score},"
    )

        shipment.agent_trace.append(f"\n{trace_entry}\n")
    
        return shipment
    
