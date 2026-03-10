import pytest
from unittest.mock import MagicMock
import json
from src.agents.agents.document_processor import DocumentAgent
from src.models.data_models import ShipmentModel

def test_document_agent_process_success():
    mock_llm = MagicMock()
    
    # Use the keys that DocumentConverter.to_shipment actually looks for!
    fake_llm_json = {
        "waybill_id": "WAYBILL-999",       # Match: raw_extraction.get("waybill_id")
        "pickup_location": "123 Apple St", # Match: raw_extraction.get("pickup_location")
        "delivery_location": "456 Orange", # Match: raw_extraction.get("delivery_location")
        "quantity": 2,                     # Match: raw_extraction.get("quantity")
        "total_weight": 5.5,               # Match: raw_extraction.get("total_weight")
        "category": "Electronics",         # Match: raw_extraction.get("category")
        "pickup_date_time": "2026-03-09T10:00:00" # ISO format
    }
    
    mock_response = MagicMock()
    mock_response.content = json.dumps(fake_llm_json)
    mock_llm.invoke.return_value = mock_response

    agent = DocumentAgent(llm_client=mock_llm)
    result = agent.process("Dummy text")

    # This will now pass!
    assert result.shipment_id == "WAYBILL-999"
    assert result.total_weight_kg == 5.5
    # Note: in your model, parcel_count has an alias to passenger_count
    assert result.parcel_count == 2

def test_document_agent_with_feedback():
    # SETUP: Mock LLM again
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({"shipment_id": "REVISED-1", "total_weight_kg": 10.0})
    mock_llm.invoke.return_value = mock_response

    agent = DocumentAgent(llm_client=mock_llm)
    
    # ACT: Pass feedback to the agent
    agent.process("waybill text", feedback="Weight was wrong")

    # ASSERT: Check if the feedback was included in the prompt sent to LLM
    called_args = mock_llm.invoke.call_args[0][0] # Get the messages list
    user_message = called_args[1][1] # Get the second message (user) content
    
    assert "REVISION REQUEST" in user_message
    assert "Weight was wrong" in user_message