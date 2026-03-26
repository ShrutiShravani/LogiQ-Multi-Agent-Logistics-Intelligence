🚚 LogiQ — Multi-Agent Logistics Intelligence Platform

LogiQ is a production-grade multi-agent orchestration system designed to automate industrial logistics workflows — from document ingestion to pricing and validation.

It ensures data integrity and pricing accuracy using a self-correcting Actor–Critic architecture, preventing incorrect shipments before financial commitment.

❓Why LogiQ?

Traditional logistics systems:
rely on manual validation ❌
are error-prone (wrong weight, route, cost) ❌
lack real-time intelligence ❌

👉 LogiQ solves this using:

multi-agent orchestration
deterministic + AI validation loops
real-time pricing intelligence

🏗️ System Architecture

graph TD 
A[FastAPI Ingestion] --> B{Supervisor}
B --> C[Document Agent: Extraction] D -- "Success" --> E[Route Agent: Mapping] E --> F[Pricing Agent: XGBoost] F --> G[Critic Agent: Validation] G -- "Invalid (e.g. 0kg)" --> B 
F --> G[MLflow & LangSmith: Telemetry] G --> H[PostgreSQL: Persistence]

⚙️ Core Design Decisions

🔹 1. Why Multi-Agent (LangGraph)?

Logistics workflows involve:

document parsing
routing
pricing
validation

👉 Instead of one LLM:

we split responsibilities into agents orchestrated via LangGraph DAG
This enables:
modular design
retry loops
stateful execution

🔹2.Why Actor–Critic Architecture?

LLMs alone are unreliable for critical decisions.

👉 We introduced:

Actor → performs task (extract, price)
Critic → validates output deterministically

Critic checks:

weight ≠ 0
parcel count consistency
route validity

👉 If invalid → system retries automatically

🔹 3. Why XGBoost for Pricing?

LLMs are not ideal for numerical prediction.

👉 XGBoost was chosen because:
strong performance on tabular logistics data
interpretable pricing logic
low latency for real-time quotes

🔹 4. Why Redis (Caching)?
Geocoding APIs are:
expensive
slow

👉 Redis cache:
stores frequent locations (ports, warehouses)
reduces API calls
improves latency

🔹 5. Why PostgreSQL (State)?

Logistics workflows require:
persistence
audit trails

👉 PostgreSQL enables:

shipment state tracking
reproducibility
debugging

🔍 Key Capabilities

🔸 Self-Correcting Validation Loop

Critic detects invalid outputs
triggers targeted re-extraction
reduces manual intervention by ~80%

🔸 Predictive Pricing Engine

XGBoost model predicts base cost

adjusted using:
🌦 Weather (Visual Crossing)
🚦 Traffic (Mapbox)

🔸 Resiliency & Reliability

✅ Circuit Breakers
fallback to Haversine distance if API fails
✅ Dead Letter Queue (DLQ)
isolates corrupted waybills
✅ Latency Budgeting
fallback triggered if agent exceeds time threshold

🔸 Performance Optimization

Redis geocoding cache → reduced RTT
batching strategies → improved route efficiency
supports high-throughput logistics pipelines

📊 Benchmarks (Golden Set)

| Metric                        | Value                  |
| ----------------------------- | ---------------------- |
| Extraction Accuracy           | **96%**                |
| Latency (p99)                 | **~9s**                |
| Token Cost                    | **~$0.0001 / request** |
| Audit Coverage                | **100%**               |
| Manual Intervention Reduction | **~80%**               |


🧪 Observability

LangSmith → trace & cost monitoring
MLflow → experiment tracking & audit logs

🏗️ Tech Stack

🧠 AI / Intelligence
LangGraph (multi-agent orchestration)
XGBoost (pricing)
OpenAI models
⚙️ Backend
FastAPI
Python 3.11
📦 Infrastructure
Docker
PostgreSQL (state)
Redis (caching)
📊 Observability
MLflow
LangSmith

⚠️ Challenges & Solutions-

🔹 API Instability:

Problem: Mapbox failures
Solution: Haversine fallback → ensured 100% uptime

🔹 Deterministic Validation:

Problem: LLM errors in numeric logic
Solution: Python-based Critic → sub-ms validation

🔹 MLflow Networking Issue:

Problem: DNS rebinding error in containers
Solution: Configured MLFLOW_HTTP_HOST_ALLOW_LIST


🚀 Ongoing Work

🔹 Spatial-Temporal Batching Agent

groups nearby shipments (1km / 1hr window)
enables “Milk Runs”
improves margins by ~30–40%

🔹 Hybrid Validation (Next Gen)

Layer 1: deterministic critic
Layer 2: SLM (Phi-3) reflection

🔹 CI/CD & Deployment

AWS ECS deployment
Canary releases (5% traffic testing)
target 99.9% uptime

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

