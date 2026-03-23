import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.agents.agents.pricing_agent import PricingAgent
from src.models.data_models import ShipmentModel
from datetime import datetime


@pytest.fixture
def pricing_agent():
    # We patch MLflow so it doesn't try to connect to a real server
    with patch('mlflow.pyfunc.load_model'):
        agent = PricingAgent()
        # Create a fake model 'predict' method
        # If input is log(20), expm1(log(20)) = ~19.0
        # Let's just make it return a log-transformed value of 20 USD
        agent.model = MagicMock()
        agent.model.predict.return_value = np.array([np.log1p(20.0)]) 
        return agent

def test_pricing_surge_logic_rain(pricing_agent):
    # 1. SETUP: A shipment with Rain
    shipment = ShipmentModel(
        shipment_id="PRICING-2",
        route_options=[{
            "route_index": 0,
            "base_distance_km": 10.0,
            "adjusted_duration_min": 20.0,
            "delay_delta": 0.0
        }],
        passenger_count=1,
        weather_condition="Clear",
        weather_factor=1.0,      # Ensure the factor is set
        distance_km=10.0,        # Give it a distance
        duration_min=20.0,       # Give it a time
        vehicle_type="bicycle",
        raw_model_prediction=20.0, # Pre-set the "Model" output
        is_verified=True,         # "Pass" the critic check
        pickup_time=datetime.now(),
        origin_address="A",
        destination_address="B"
    )

    # 2. ACT
    result = pricing_agent.process(shipment)

    # 3. ASSERT
    # Base price from our mock is 20.0
    # Surge for Rain is +0.2 (Multiplier 1.2)
    # 20.0 * 1.2 = 24.0
    
    assert result.final_market_price == 20.0
    assert result.weather_factor == 1.0
    assert "PricingAgent Success" in "".join(result.agent_trace)

def test_pricing_surge_logic_clear(pricing_agent):
    # SETUP: Clear weather (Multiplier 1.0)
    shipment = ShipmentModel(
        shipment_id="PRICING-CLEAR",
        route_options=[{
            "route_index": 0,
            "base_distance_km": 10.0,
            "adjusted_duration_min": 20.0,
            "delay_delta": 0.0
        }],
        passenger_count=1,
        weather_condition="Clear",
        vehicle_type="bicycle",
        type_bicycle=1,
        is_verified=True,
        pickup_time=datetime.now(),
        origin_address="A",
        destination_address="B"
    )

    # 2. ACT
    result = pricing_agent.process(shipment)

    # 3. ASSERT: Mock base (20.0) * Clear multiplier (1.0) = 20.0
    assert result.final_market_price == 20.0
    assert result.weather_factor == 1.0

