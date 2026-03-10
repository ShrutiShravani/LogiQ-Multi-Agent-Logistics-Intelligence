import pytest
from unittest.mock import MagicMock, patch
from src.agents.agents.route_agent import RouteAgent
from src.models.data_models import ShipmentModel
from datetime import datetime

@pytest.fixture
def route_agent():
    # We use 'patch' to prevent the actual APIs from being called
    with patch('src.agents.agents.route_agent.Nominatim'), \
         patch('src.agents.agents.route_agent.Logisticscache'), \
         patch.dict('os.environ', {'MAPBOX_ACCESS_TOKEN': 'fake_test_token'}):
        agent = RouteAgent()
        return agent

def test_route_agent_process_logic(route_agent):
    # 1. SETUP: Create a shipment that needs routing
    shipment = ShipmentModel(
        shipment_id="ROUTE-101",
        origin_address="Times Square, NY",
        destination_address="Central Park, NY",
        total_weight_kg=2.5,
        passenger_count=1,
        pickup_time=datetime.now()
    )

    # 2. MOCK: Internal methods to avoid real API calls
    route_agent._get_coords = MagicMock(side_effect=[(40.7580, -73.9855), (40.7812, -73.9665)])
    route_agent.get_mapbox_route = MagicMock(return_value={
        "code": "Ok",
        "routes": [{"distance": 5000, "duration": 600}] # 5km, 10min
    })
    route_agent.get_weather_impact = MagicMock(return_value=(0.0, "Clear"))
    route_agent.get_historical_traffic = MagicMock(return_value=0.8)

    # 3. ACT
    result = route_agent.process(shipment)

    # 4. ASSERT
    assert result.distance_km == 5.0
    assert result.pickup_latitude == 40.7580
    assert result.weather_condition == "Clear"
    # Logic check: duration_min / traffic_score (600s/60 / 0.8 = 12.5)
    assert result.duration_min == 12.5