import pytest
from unittest.mock import patch, mock_open
import json
from src.agents.agents.critic_agent import CriticAgent
from src.models.data_models import ShipmentModel
from datetime import datetime

@pytest.fixture
def critic_agent():
    mock_json_content = '{"10_1_False": {"actual_speed_kmh": 20.0}}'
    
    # 1. Store the original open function
    original_open = open

    # 2. Define a replacement function that only mocks OUR file
    def mocked_open(name, *args, **kwargs):
        if "traffic_mapping.json" in str(name):
            return mock_open(read_data=mock_json_content).return_value
        return original_open(name, *args, **kwargs)
    
    with patch("builtins.open", side_effect=mocked_open):
        agent = CriticAgent()
        return agent
        
def test_critic_rejects_zero_weight(critic_agent):
    # 1. SETUP: Weight is 0.0 (Invalid)
    shipment = ShipmentModel(
        shipment_id="TEST-001",
        total_weight_kg=0.0,
        passenger_count=1,
        origin_address="A",
        destination_address="B",
        pickup_time=datetime.now()
    )

    # 2. ACT
    result = critic_agent.process(shipment)

    # 3. ASSERT
    assert result.is_verified is False
    assert "CRITIC_REJECT" in "".join(result.agent_trace)
    assert "Weight is 0.0 or missing" in "".join(result.agent_trace)

def test_critic_overrides_wrong_vehicle(critic_agent):
    # 1. SETUP: 5km distance, 2kg weight should be 'bicycle' 
    # but we will manually set it to 'truck'
    shipment = ShipmentModel(
        shipment_id="TEST-002",
        total_weight_kg=2.0,
        distance_km=5.0,
        vehicle_type="truck", # MISTAKE!
        passenger_count=1,
        origin_address="A",
        destination_address="B",
        pickup_time=datetime.now(),
        hour=10,
        day_of_week=1,
        is_holiday=False
    )

    # 2. ACT
    result = critic_agent.process(shipment)

    # 3. ASSERT
    assert result.vehicle_type == "bicycle" # Corrected!
    assert "Vehicle: truck -> bicycle" in "".join(result.agent_trace)
    assert result.type_bicycle == 1
    assert result.type_truck == 0

def test_critic_corrects_insane_pricing(critic_agent):
    # 1. SETUP: Set a very low price manually
    shipment = ShipmentModel(
        shipment_id="TEST-003",
        total_weight_kg=5.0,
        distance_km=5.0,
        vehicle_type="bicycle",
        raw_model_prediction=1.0, # Way too cheap!
        predicted_base_price=1.0,
        weather_factor=1.0,
        passenger_count=1,
        origin_address="A",
        destination_address="B",
        pickup_time=datetime.now(),
        hour=10,
        day_of_week=1,
        is_holiday=False,
        duration_min=15.0,
        traffic_density_score=1.0,
        extraction_attempts=0,
    )

    # 2. ACT
    result = critic_agent.process(shipment)

    # 3. ASSERT
    # Theoretical price for bicycle: base (3.0) + 5km*1.0 + 1*1.5 + 15*0.2 = 12.5
    assert result.predicted_base_price == 12.5 
    assert result.is_verified is True
    assert "Price: $1.0 -> $12.5" in "".join(result.agent_trace)