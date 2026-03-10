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
        shipment_id="PRICING-1",
        passenger_count=1,
        total_weight_kg=5.0,
        distance_km=10.0,
        weather_condition="Rain", # Should trigger +0.2 surge
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
    assert result.predicted_base_price == 20.0
    assert result.final_market_price == 24.0
    assert result.weather_factor == 1.2
    assert "PricingAgent Success" in "".join(result.agent_trace)

def test_pricing_surge_logic_clear(pricing_agent):
    # SETUP: Clear weather (Multiplier 1.0)
    shipment = ShipmentModel(
        shipment_id="PRICING-2",
        passenger_count=1,
        weather_condition="Clear",
        pickup_time=datetime.now(),
        origin_address="A",
        destination_address="B"
    )

    # ACT
    result = pricing_agent.process(shipment)

    # ASSERT: Price should stay 20.0
    assert result.final_market_price == 20.0
    assert result.weather_factor == 1.0