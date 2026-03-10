import pytest
from src.agents.agents.orchestrator import create_logisticsgraph
from src.models.data_models import ShipmentModel
from unittest.mock import MagicMock, patch
from datetime import datetime

@pytest.fixture
def mock_agents():
    return {
        "doc": MagicMock(),
        "route": MagicMock(),
        "pricing": MagicMock(),
        "critic": MagicMock()
    }

def test_full_graph_flow_success(mock_agents):
    # 1. SETUP: Mock the agents to return valid data
    app = create_logisticsgraph(
        mock_agents["doc"], 
        mock_agents["route"], 
        mock_agents["pricing"], 
        mock_agents["critic"]
    )
    
    # Mock Document Agent Output
    shipment = ShipmentModel(
        shipment_id="INT-001", passenger_count=1, origin_address="A", 
        destination_address="B", pickup_time=datetime.now(), total_weight_kg=5.0
    )
    mock_agents["doc"].process.return_value = shipment
    
    # Mock Route Agent (Add coords so it doesn't end early)
    def add_coords(s):
        s.pickup_latitude, s.dropoff_latitude = 40.7, 40.8
        return s
    mock_agents["route"].process.side_effect = add_coords
    
    # Mock Pricing and Critic (Success)
    mock_agents["pricing"].process.return_value = shipment
    shipment.is_verified = True
    mock_agents["critic"].process.return_value = shipment

    # 2. ACT
    initial_state = {
        "waybill_text": "Dummy text",
        "shipment": None,
        "attempts": 0,
        "error_log": []
    }
    final_state = app.invoke(initial_state)

    # 3. ASSERT
    assert final_state["shipment"].is_verified is True
    assert final_state["attempts"] == 1
    # Verify the chain was called in order
    mock_agents["doc"].process.assert_called_once()
    mock_agents["critic"].process.assert_called_once()

def test_graph_retry_on_critic_fail(mock_agents):
    app = create_logisticsgraph(
        mock_agents["doc"], mock_agents["route"], 
        mock_agents["pricing"], mock_agents["critic"]
    )

    # Mock Critic to fail the first time, then succeed
    fail_shipment = ShipmentModel(
        shipment_id="RETRY-1", passenger_count=1, origin_address="A", 
        destination_address="B", pickup_time=datetime.now(), is_verified=False
    )
    fail_shipment.agent_trace.append("CRITIC_FEEDBACK: Missing weight")
    
    success_shipment = fail_shipment.model_copy()
    success_shipment.is_verified = True

    mock_agents["doc"].process.return_value = fail_shipment
    mock_agents["route"].process.return_value = fail_shipment
    mock_agents["pricing"].process.return_value = fail_shipment
    
    # First call returns fail, second returns success
    mock_agents["critic"].process.side_effect = [fail_shipment, success_shipment]
    mock_agents["doc"].process.side_effect = [fail_shipment, success_shipment]

    # ACT
    final_state = app.invoke({"waybill_text": "...", "shipment": None, "attempts": 0, "error_log": []})

    # ASSERT
    assert final_state["attempts"] == 2  # It retried!
    assert "Missing weight" in final_state["error_log"][0]