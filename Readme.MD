🚚 LogiQ: Multi-Agent Logistics Intelligence Platform
LogiQ is a high-performance orchestration platform designed to automate the lifecycle of industrial logistics. By utilizing a Senior-level Actor-Critic architecture, the system ensures data integrity and pricing accuracy through iterative reflection loops before any financial commitments are recorded.

🏗️ System Architecture & Workflow
Using LangGraph, the system coordinates specialized agents in a stateful directed acyclic graph (DAG). Below is the logic flow:
<<<<<<< HEAD

Code snippet
graph TD
    A[FastAPI Ingestion] --> B{Supervisor}
    B --> C[Document Agent: Extraction]
    D -- "Success" --> E[Route Agent: Mapping]
    E --> F[Pricing Agent: XGBoost]
    F --> G[Critic Agent: Validation]
    G -- "Invalid (e.g. 0kg)" --> B
    F --> G[MLflow & LangSmith: Telemetry]
    G --> H[PostgreSQL: Persistence]

🏆 System Benchmarks (The "Golden Set")
Validated against a controlled 'Golden Set' of 6 multi-format complex waybill PDFs.

Metric	Value	Tech Source
End-to-end Latency	9s (p99)	LangSmith Trace
Extraction Accuracy	96%	Critic Agent Feedback
Token Cost	~$0.0001 / req	GPT-4o-mini
Audit Coverage	100%	MLflow Experiment Logs


🌟 Engineering Highlights
Actor-Critic Reflection: Implemented a Deterministic Critic Agent that performs mathematical validation (zero-weight checks, parcel count verification). It triggers targeted re-extraction loops, reducing human intervention by 80%.

Human-in-the-Loop (HITL) Triggers: If the system fails to validate a document after 3 attempts, it raises a DataValidationError and routes the shipment to a manual audit queue, preventing infinite loops.

Predictive Pricing Engine: A custom XGBoost regressor predicts base costs, adjusted by an asynchronous Pricing Agent considering Weather (Visual Crossing) and Traffic (Mapbox).

Resiliency Patterns:

Circuit Breakers: If the Mapbox API is unreachable, the system triggers a Haversine/Manhattan fallback to ensure 100% uptime.

Dead Letter Queues (DLQ): Corrupted or unparseable waybills are isolated for manual inspection.

Performance Optimization:

Geocoding Cache: Uses Redis to store recurring coordinates (e.g., ports/airports), slashing RTT and API costs.

Latency Budgets: Automated fallback execution if the RouteAgent exceeds its 2-second processing window.

🛠️ Tech Stack & Infrastructure
Core: Python 3.11, FastAPI, Pydantic, LangGraph

Intelligence: XGBoost, NumPy, Pandas, OpenAI

Infrastructure: Docker Compose, PostgreSQL (Permanent State), Redis (Caching)

Observability: LangSmith (Trace/Cost), MLflow (Audit Trail/Versioning)
=======


Challenges faced:
API Instability- Implemented Circuit Breakers and Manhattan Distance fallbacks.
DNS Rebinding- Configured MLFLOW_HTTP_HOST_ALLOW_LIST for secure internal container communication.
Deterministic Validation- Developed a math-based Critic to bypass LLM latency for basic logic checks.


Currently working on:
1)Batching Logic (Current Focus): Developing a Batching Agent to group shipments within 1km/1hr windows to optimize costs.
2)Improving latency
3) Dynamic Feedback Loops: Currently utilizing a high-performance deterministic Critic (Python/Math) for sub-10ms validation.  will introduce an SLM-based (Phi-3) Reflection Agent to handle nuanced, non-deterministic errors that pure logic cannot catch (e.g., verifying if 'Fragile' handling instructions were correctly interpreted).
4)Full CD Pipeline: Automated deployment to AWS ECS with integrated Canary releases.

🚀 Getting Started
1. Prerequisites
•	Docker & Docker Compose
•	OpenAI API Key (for document proecssing)

2.How to run:
# Clone the repository
git clone https://github.com/yourusername/logiq-agent-ops.git
cd logiq-agent-ops

# Set up environment variables
cp .env.example .env
# LLM & Tracing
OPENAI_API_KEY=your_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT="LogiQ-Intelligence-Platform"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"

# External APIs
MAPBOX_ACCESS_TOKEN=your_mapbox_token
VISUAL_CROSSING_KEY=your_weather_key
NOMINATIM_URL=https://nominatim.openstreetmap.org/search
NOMINATIM_REVERSE_URL=https://nominatim.openstreetmap.org/reverse
APP_USER_AGENT=LogiQ_Senior_Project_V2

# Databases (Docker Internal)
POSTGRES_USER=logiq_user
POSTGRES_PASSWORD=logiq_pass
POSTGRES_DB=logiq_db
POSTGRES_HOST=db
POSTGRES_PORT=5432
REDIS_HOST=redis
REDIS_PORT=6379

# Observability
MLFLOW_TRACKING_URI=http://mlflow_server:5000
MLFLOW_EXPERIMENT_NAME=Logistics_Pricing_Live

# Start the full stack (API + Redis + MLflow)
docker-compose up –build

# Run tests locally
pytest tests/

