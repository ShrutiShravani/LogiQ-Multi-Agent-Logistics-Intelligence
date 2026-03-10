from src.agents.agents.critic_agent import CriticAgent
from src.agents.agents.document_processor import DocumentAgent
from src.agents.agents.pricing_agent import PricingAgent
from src.agents.agents.route_agent import RouteAgent
import fitz
from src.agents.agents.orchestrator import create_logisticsgraph
from openai import OpenAI
import os
import json
import mlflow
import glob
from langchain_openai import ChatOpenAI
from datetime import datetime


def extract_text(pdf_path):
        """Utility to turn the PDF file into a string the LLM can read"""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

if __name__=="__main__":
    # 1. Initialize Agents
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    doc_agent = DocumentAgent(llm_client=llm)
    route_agent = RouteAgent()
    pricing_agent = PricingAgent()
    critic_agent = CriticAgent()
    input_path= 'data/raw'
    
 
    pdf_files = glob.glob(os.path.join(input_path,"*.pdf"))
    
    stats_for_metrics = []
    all_shipment_data=[]
    all_audit_traces={}
    mlflow.set_experiment("Logistics_Pricing")
    
    # 2. Build the Graph
    with mlflow.start_run(run_name="7_Waybill_Stress_Test"):
        app = create_logisticsgraph(doc_agent, route_agent, pricing_agent, critic_agent)
       
        if not pdf_files:
            print(f"No files found in {input_path}")
        else:
            for pdf_path in pdf_files:
                results=[]
            
                waybill_text = extract_text(pdf_path)
                initial_state = {
                "waybill_text": waybill_text,
                "shipment": None,       
                "feedback": None,       
                "attempts": 0,        
                "error_log": []      
                }

                # 4. Execute
                final_state = app.invoke(initial_state)
                shipment = final_state['shipment']

                #calcualte theoretical price
                theory_price = critic_agent._calculate_theoretical_price(shipment)

                stats_for_metrics.append({
                    "pred": shipment.raw_model_prediction,
                    "actual_price": theory_price,
                    "overridden": any("CriticAgent:"in trace and "Success" not in trace for trace in shipment.agent_trace)
                })
                # Define the keys we want to extract from the Pydantic object
                keys_to_show = { 
                    "shipment_id",
                    "origin_address",
                    "destination_address",
                    "total_weight_kg",
                    "item_category",
                    "parcel_count",
                    "pickup_time",
                    "pickup_latitude",
                    "pickup_longitude",
                    "dropoff_latitude",
                    "dropoff_longitude",
                    "hour",
                    "day_of_week",
                    "is_rush_hour",
                    "is_holiday",  
                    "is_weekend",
                    "is_high_demand",
                    "traffic_density_score",
                    "distance_km",
                    "duration_min",
                    "predicted_base_price",
                    "final_market_price", 
                    "vehicle_type", 
                    "weather_condition",
                    "is_verified", 
                }

                # 5. Extract and Transform
                # Using a dictionary comprehension to pull data from the ShipmentModel object
                final_output = {k: getattr(shipment, k, "N/A") for k in keys_to_show}

                # Add Metadata & Formatting
                final_output["currency"] = "USD"

                # Safety check for duration_min before popping
                duration = final_output.pop("duration_min", 0)
                final_output["eta_minutes"] = duration if duration != "N/A" else 0

                final_output["status"] = "SUCCESS" if getattr(shipment, "is_verified", False) else "FAILED"
                final_output["total_retries"] = final_state.get("attempts", 1) - 1
                final_output["system_errors"] = final_state.get("error_log", [])

                # 6. Route to DLQ or Results
                if not shipment.is_verified:
                    review_entry = {
                        "shipment_id": shipment.shipment_id,
                        "failure_trace": shipment.agent_trace,
                        "raw_text": waybill_text,
                        "system_errors": final_output["system_errors"]
                    }
                    # Save DLQ immediately
                    with open("data/processed/manual_review.json", "a") as f:
                        f.write(json.dumps(review_entry) + "\n")
                    mlflow.log_metric("manual_intervention_required", 1)
                else: 
                    all_shipment_data.append(final_output)
                    all_audit_traces[shipment.shipment_id] = shipment.agent_trace

            #calcualte mae
            MAE= sum(abs(r['pred']-r['actual_price']) for r in stats_for_metrics)/len(stats_for_metrics)

            #calcualte oevrrides count
            overrides_count= sum(1 for r in stats_for_metrics if r['overridden'])
            print(f"override_count:{overrides_count}")
            print(len(stats_for_metrics))
            overrides_rate= (overrides_count/len(stats_for_metrics))* 100

            if overrides_rate>30.0:
                mlflow.set_tag("alert", "Potential_Model_Drift_Detected")
                print("WARNING: High override rate detected. Model may require retraining.")

        
            #log to mlflow
            mlflow.log_metric("MAE_PRICE", round(MAE, 2))
            mlflow.log_metric("Critic_Override_Percent",round(overrides_rate,2))
    
            mlflow.log_param("total_test_cases", len(pdf_files))
          
            mlflow.log_param("model_version","xgboost_logistics_pricing_Version_1")

            print(f"test_complete:{MAE},Override rate:{overrides_rate}%")

        
            with open("batch_results.json", "w") as f:
                json.dump(all_shipment_data, f, indent=4,default=json_serial)
            with open("batch_full_audit.json","w") as f:
                json.dump(all_audit_traces,f,indent=4)

