LogiQ — Multi-Agent Logistics Intelligence Platform

Production-grade AI system for automated logistics validation, pricing, and decision intelligence.

LogiQ is a multi-agent orchestration platform that automates end-to-end logistics workflows — from document ingestion to route validation and pricing.

It introduces a self-correcting Actor–Critic architecture to ensure zero-invalid shipments before financial commitment, solving one of the most critical failure points in logistics systems.

Problem

Modern logistics systems suffer from:

manual validation workflows
high error rates (incorrect weight, routing, pricing)
lack of real-time intelligence
costly downstream failures

Even a single incorrect shipment can lead to financial loss, SLA violations, and operational inefficiencies.

Solution

LogiQ replaces brittle pipelines with a multi-agent, self-correcting system:

decomposes workflows into specialized agents
validates outputs deterministically using a Critic layer
retries automatically on failure
integrates ML + rules for reliable decision-making

Result: high accuracy, lower cost, and production reliability

Architecture Overview

graph TD 
A[FastAPI Ingestion] --> B{Supervisor}
B --> C[Document Agent: Extraction]
C --> D{Validation}
D -- "Success" --> E[Route Agent: Mapping]
E --> F[Pricing Agent: XGBoost]
F --> G[Critic Agent: Validation]
G -- "Invalid (e.g. 0kg)" --> B
F --> I[MLflow & LangSmith: Telemetry]
G --> H[PostgreSQL: Persistence]

Core Design Decisions

1. Why Multi-Agent (LangGraph)?

Logistics workflows involve:

document parsing
routing
pricing
validation

Instead of one LLM:

responsibilities are split into agents orchestrated via a LangGraph DAG

This enables:

modular design
retry loops
stateful execution


2. Why Actor–Critic Architecture?

LLMs alone are unreliable for critical decisions.

We introduced:

Actor → performs task (extract, price)
Critic → validates output deterministically

Critic checks:

weight ≠ 0
parcel count consistency
route validity

If invalid, the system retries automatically.

3. Why XGBoost for Pricing?

LLMs are not ideal for numerical prediction.

XGBoost was chosen because:

strong performance on tabular logistics data
interpretable pricing logic
low latency for real-time quotes


4. Why Redis (Caching)?

Geocoding APIs are:

expensive
slow

Redis cache:

stores frequent locations (ports, warehouses)
reduces API calls
improves latency


5. Why PostgreSQL (State)?

Logistics workflows require:

persistence
audit trails

PostgreSQL enables:

shipment state tracking
reproducibility
debugging


Key Innovations

1. Actor–Critic Validation Loop

Traditional systems trust model outputs. LogiQ does not.

Actor → performs task (extraction, pricing)
Critic → validates using deterministic rules

Validation includes:

weight integrity (non-zero, consistent)
parcel count checks
route feasibility

If validation fails:

system re-triggers upstream agents
no invalid output reaches final stage

Impact:

~80% reduction in manual intervention
near-zero critical shipment errors

2. Multi-Agent Orchestration (LangGraph)

Instead of a monolithic LLM:

each step is an independent agent
orchestrated via DAG execution
supports retries and state persistence

Benefits:

modular system design
fault isolation
scalable workflows
3. Hybrid Intelligence (ML + Systems)

LogiQ combines:

LLMs → extraction and reasoning
XGBoost → pricing prediction
deterministic rules → validation

Why this matters:

avoids LLM hallucination in critical paths
ensures reliability in production

4. Real-Time Pricing Engine

Pricing is computed using:

XGBoost base model (tabular features)
real-time adjustments:
weather data (Visual Crossing)
traffic data (Mapbox)

Result:

accurate and explainable pricing
low-latency predictions

5. High-Performance Infrastructure
Redis caching → reduces geocoding latency
PostgreSQL → persistent state and auditability
asynchronous agent execution → improved throughput

System Capabilities

Self-Correcting Execution-
automatic retries on validation failure
targeted reprocessing instead of full restart

Resiliency Engineering-

circuit breakers for API failures
Haversine fallback for routing
dead letter queue for corrupted inputs
latency-based fallback triggers

Performance Optimization-
caching frequently used geolocations
batching route computations
optimized for high-throughput pipelines


Benchmarks

| Metric                        | Value              |
| ----------------------------- | ------------------ |
| Extraction Accuracy           | 96%                |
| Latency (p99)                 | ~9s                |
| Token Cost                    | ~$0.0001 / request |
| Audit Coverage                | 100%               |
| Manual Intervention Reduction | ~80%               |


Observability

LangSmith → tracing, debugging, cost monitoring
MLflow → experiment tracking and audit logs


full system visibility
reproducible pipelines
production debugging


Tech Stack

AI and Intelligence
LangGraph (multi-agent orchestration)
OpenAI models
XGBoost
Backend
FastAPI
Python 3.11
Infrastructure
Docker
PostgreSQL
Redis
Observability
MLflow
LangSmith


Challenges and Engineering Decisions

API Reliability-

Problem: external API failures (Mapbox)
Solution: fallback to Haversine distance → ensured 100% uptime

LLM Reliability-

Problem: incorrect numerical outputs
Solution: deterministic Critic layer → sub-millisecond validation

Distributed System Issues-

Problem: MLflow DNS rebinding in containers
Solution: configured host allowlist

Currently Working On/ Optimizations Sprint-

Spatial-Temporal Optimization
cluster shipments within 1 km / 1 hour windows
enables "Milk Runs"
expected margin improvement: 30–40%

Hybrid Validation Layer
Layer 1: deterministic validation
Layer 2: small model (Phi-3) reflection

Production Deployment
AWS ECS deployment
canary releases (5% traffic)
target: 99.9% uptime

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

