from src.agents.agents.critic_agent import CriticAgent
from src.agents.agents.document_processor import DocumentAgent
from src.agents.agents.pricing_agent import PricingAgent
from src.agents.agents.route_agent import RouteAgent
import fitz
from src.agents.agents.orchestrator import create_logisticsgraph
from dotenv import load_dotenv
import os
import json
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import time
import random
from datetime import datetime
import mlflow
from collections import deque
import numpy as np

load_dotenv()
override_window = deque(maxlen=100)
failure_window = deque(maxlen=100)
app = FastAPI(title="LogiQ Logistics Intelligence API")


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


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000")
if MLFLOW_TRACKING_URI != "local":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Logistics_API_Live")

def extract_text(content:bytes):
    """Utility to turn the PDF file into a string the LLM can read"""
    doc = fitz.open(stream=content,filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

logistics_app = create_logisticsgraph(doc_agent, route_agent, pricing_agent, critic_agent)

#api endpoints
@app.get("/health")
def healthcheck():
    return {"status":"healthy","model":"gpt-40-mini"}
@app.get("/")
def home():
    return {"status": "success", "message": "Pricing API is running without MLflow!"}

@app.post("/process-waybill")
async def process_Waybill(file:UploadFile=File(...)):
    """
    Accepts a PDF file, runs the full agentic graph, and returns structured data.
    """
    start_time = time.time()
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400,detail="Only pdf files s are supported.")

    try:
        pdf_bytes = await file.read()
        #read file content
        waybill_text= extract_text(pdf_bytes)

        #intialize state
        initial_state={
                "waybill_text": waybill_text,
                "shipment": None,       
                "feedback": None,       
                "attempts": 0,        
                "error_log": []      
        }

        final_state = logistics_app.invoke(initial_state)
        shipment = final_state['shipment']

        # --- 3. METRIC CALCULATIONS ---
        latency = time.time() - start_time

        # Exact override logic from your main.py
        is_overridden = any("CriticAgent:" in trace and "Success" not in trace for trace in shipment.agent_trace)
        is_verified = getattr(shipment, "is_verified", False)

        # Update Sliding Windows
        override_window.append(1 if is_overridden else 0)
        failure_window.append(0 if is_verified else 1)

        # Calculate Rates
        ov_rate = (sum(override_window) / len(override_window)) * 100
        fail_rate = (sum(failure_window) / len(failure_window)) * 100

        #mlflow logging and alerts
        with mlflow.start_run(run_name=f"Shipment_{shipment.shipment_id}", nested=True):
            mlflow.log_metric("latency", latency)
            mlflow.log_metric("live_override_rate", ov_rate)
            mlflow.log_metric("live_failure_rate", fail_rate)


            if len(override_window) >= 20 and ov_rate > 15.0:
                mlflow.set_tag("drift_alert", "HIGH_OVERRIDE_RATE")
            
            if len(failure_window) >= 20 and fail_rate > 10.0:
                mlflow.set_tag("system_alert", "HIGH_FAILURE_RATE")

        if not is_verified:
            review_entry = {
                "shipment_id": shipment.shipment_id,
                "failure_trace": shipment.agent_trace,
                "raw_text": waybill_text
            }
            # Save DLQ immediately
            os.makedirs("data/processed", exist_ok=True)
            with open("data/processed/manual_review.json", "a") as f:
                f.write(json.dumps(review_entry) + "\n")
          

        keys_to_show = { 
                    "shipment_id",
                    "origin_address",
                    "destination_address",
                    "total_weight_kg",
                    "item_category",
                    "parcel_count",
                    "pickup_time",
                    "distance_km",
                    "duration_min",
                    "vehicle_type",
                    "final_market_price", 
                    
                }
        
        output={}
        for k in keys_to_show:
            val = getattr(shipment, k, "N/A")

            # Use your logic: convert NumPy types to Python types
            if isinstance(val, (np.float32, np.float64)):
                val = float(val)
            elif isinstance(val, (np.int32, np.int64)):
                val = int(val)
            elif isinstance(val, datetime):
                val = val.isoformat()
                
            output[k] = val
    
        output["status"]="SUCCESS" if shipment.is_verified else "Failed"
        output["metrics"] = {
            "latency_sec": round(latency, 2),
            "current_window_drift": f"{round(ov_rate, 2)}%",
            "current_window_failure": f"{round(fail_rate, 2)}%"
        }
        return output
    
    except  Exception as e:
        with mlflow.start_run(run_name="API_CRASH"):
            mlflow.log_param("error", str(e))
        raise HTTPException(status_code=500, detail=str(e))
        
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

        




