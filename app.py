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
from src.models.data_models import ShipmentModel
import uvicorn

load_dotenv()

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
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400,detail="Only pdf files s are supported.")

    try:
        pdf_bytes = await file.read()
        #read file content
        waybill_text= extract_text(pdf_bytes)

        #intialzie state
        initial_state={
                "waybill_text": waybill_text,
                "shipment": None,       
                "feedback": None,       
                "attempts": 0,        
                "error_log": []      
        }

        final_state = logistics_app.invoke(initial_state)
        shipment = final_state['shipment']

        if not shipment.is_verified:
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
        
        output ={k:getattr(shipment,k,"N/A") for k in keys_to_show}
        output["status"]="SUCCESS" if shipment.is_verified else "Failed"
        return output
    
    except  Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

        




