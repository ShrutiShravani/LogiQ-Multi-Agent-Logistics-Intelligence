from typing import TypedDict, List, Optional
from src.models.data_models import ShipmentModel
from langgraph.graph import StateGraph, END
from geopy.geocoders import Nominatim
import mlflow

#mlflow.set_tracking_uri("http://localhost:5000") # or your local path
#mlflow.set_experiment("Agentic_Pricing_Audit") #this sets only for local run

class AgentState(TypedDict):
    waybill_text: str
    shipment: Optional[ShipmentModel]
    feedback: Optional[str]
    attempts: int 
    error_log: List[str] 

geolocator = Nominatim(user_agent="nyc_logistics_auditor")
def create_logisticsgraph(doc_agent,route_agent,pricing_agent,critic_agent):
    workflow=StateGraph(AgentState)
 
    #document extraction
    def call_doc(state):
        shipment=doc_agent.process(state["waybill_text"], feedback=state.get("feedback"))
        # DEBUG PRINTS
        print(f"DEBUG: Raw Extracted Pickup adress: {shipment.origin_address} ")
        print(f"Raw Extracted Delivery adress: {shipment.destination_address}")
        print(f"weight: {shipment.total_weight_kg} ,parcel_count: {shipment.parcel_count}")
        print(f"pickup_time: {shipment.pickup_time}")
        return {"shipment": shipment, "attempts": state["attempts"] + 1}

    #routing and pricing
    def call_routing(state):
        shipment = state["shipment"] # Get it from the state first!
        shipment = route_agent.process(shipment)
        return {"shipment": shipment}
       
        #print(f"DEBUG: Raw Extracted Pickup Lat: {shipment.pickup_latitude} {shipment.pickup_longitude}")
       # print(f"DEBUG: Raw Extracted Pickup Lon: {shipment.dropoff_latitude} {shipment.dropoff_longitude}")
        #print(f"DEBUG: Distance calculated: {shipment.distance_km} km")
    
    def call_logistics_pricing(state):
        shipment = state["shipment"] 
        shipment = pricing_agent.process(shipment)
        #print(f"DEBUG: Final Price predicted: ${shipment.final_market_price}")
        return {"shipment": shipment}
    
    #call critic
    def call_critic(state):
        shipment = critic_agent.process(state["shipment"])
        if not state['shipment'].is_verified:
            feedback_list=[t for t in shipment.agent_trace if "CRITIC_FEEDBACK" in t]
            current_feedback = feedback_list[-1] if feedback_list else "Invalid data detected."
            old_log = state.get("error_log", [])
            new_log = old_log + [current_feedback]
        
            return {"shipment": shipment, "feedback": current_feedback,"error_log":new_log}
        return {"shipment": shipment, "feedback": None}

    #decdie  whther to continue to end loop
    def check_geo_coords(state):
        shipment = state['shipment']
        if  shipment.pickup_latitude is None or shipment.dropoff_latitude is None:
            return "end"
        return "continue"

    def should_continue(state):
        shipment = state['shipment']
        if shipment.is_verified or state["attempts"]>=3:
            return "end"
        return "retry"
    
    # Define the Graph
    workflow.add_node("document_agent", call_doc)
    workflow.add_node("route_engine", call_routing)
    workflow.add_node("pricing_engine", call_logistics_pricing)
    workflow.add_node("critic_agent", call_critic)

    workflow.set_entry_point("document_agent")
    workflow.add_edge("document_agent", "route_engine")
    workflow.add_edge("pricing_engine","critic_agent")

    workflow.add_conditional_edges(
        "critic_agent",
        should_continue,
        {
            "end": END,
            "retry": "document_agent"
        }
    )
    workflow.add_conditional_edges(
        "route_engine",
        check_geo_coords,
        {
            "end": END,           # This triggers the DLQ in your main.py loop
            "continue": "pricing_engine"
        }
    )

    return workflow.compile()