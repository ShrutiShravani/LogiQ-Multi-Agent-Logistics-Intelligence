import pandas as pd

# 1. CREATE THE CUSTOM ERROR CLASS HERE
class DataValidationError(Exception):
    """Raised Feature Vector is invalid or missing columns."""
    pass

def validate_columns(df,req_cols=None):
    if req_cols is None:
        req_cols = (
            'passenger_count', 'pickup_longitude', 'pickup_latitude', 
            'dropoff_longitude', 'dropoff_latitude', 'total_weight_kg', 
            'distance_km', 'hour', 'day_of_week', 'is_holiday', 
            'duration_min', 'traffic_density_score', 'is_rush_hour', 
            'is_weekend', 'is_high_demand', 'type_bicycle', 
            'type_e_scooter', 'type_truck', 'type_van'
        )

    for col in req_cols:
        if col not in df.columns:
            # Instead of just printing, we can raise the error here or return False
            print(f"CRITICAL: Missing column detected: {col}")
            return False
    return True

def validate_dataype(df):
    type_map = {
        'passenger_count': float, 'pickup_longitude': float, 'pickup_latitude': float,
        'dropoff_longitude': float, 'dropoff_latitude': float, 'total_weight_kg': float,
        'distance_km': float, 'hour': float, 'day_of_week': float, 'is_holiday': float,
        'duration_min': float, 'traffic_density_score': float, 'is_rush_hour': float,
        'is_weekend': float, 'is_high_demand': float, 'type_bicycle': float,
        'type_e_scooter': float, 'type_truck': float, 'type_van': float
    }

    for col, expected_type in type_map.items(): # Added parentheses to .items()
        try:
            # Senior Move: pd.to_numeric is safer than .astype for validation
            df[col] = pd.to_numeric(df[col], errors='raise').astype(expected_type)
        except Exception as e:
            print(f"CRITICAL: Column '{col}' failed type validation. Expected {expected_type}.")
            return False
    
    return True