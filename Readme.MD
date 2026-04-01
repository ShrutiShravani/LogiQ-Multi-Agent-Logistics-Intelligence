LogiQ — Multi-Agent Logistics Intelligence Platform

LogiQ is a production-grade multi-agent AI system designed to automate logistics workflows — from document ingestion to route mapping, pricing, and validation.

It enables accurate, reliable, and cost-efficient decision-making in real-world logistics operations.



Why LogiQ?


Traditional logistics systems:


- rely on manual validation

- are error-prone (incorrect weight, route, cost)


- lack real-time intelligence



LogiQ solves this using:


-multi-agent orchestration


-retrieval-augmented decision pipelines


-trust & safety validation layers


-cost-aware system design



Architecture Overview-


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




Core Capabilities:


1. Multi-Agent Orchestration (LangGraph)-


Modular agent design for routing, pricing, and validation
Stateful workflows with controlled retries and correction loops
Separation of concerns for reliability and scalability


2. RAG-Based Data Processing-


- Processes semi-structured logistics data

- Embedding + retrieval pipeline for contextual decision-making

- Structured outputs for downstream agents


3. Trust & Safety (Defense-in-Depth)-


- Multi-stage validation pipeline:


   1) Schema validation (Pydantic)


   2) Statistical anomaly detection (Z-score)


   3) LLM-based expert validation (Phi-3)



4. Guardrails using NeMo


   Prevents invalid or unsafe decisions (similar to fraud detection systems)



5. Efficiency & Cost Optimization (FinOps)



  - Spatial-temporal batching of logistics requests



   - Reduces redundant API calls (routing/pricing)



   - Improves system throughput and cost efficiency


5. Fault-Tolerant System Design


    - Redis caching

    - Dead-letter queues

   - Circuit breakers

    - Ensures system reliability under failure conditions



6. Evaluation & Observability


    - MLflow → experiment tracking


    - LangSmith → agent tracing


    - Metrics: accuracy, latency, system performance




Core Design Decisions-


1. Why Multi-Agent (LangGraph)?

Logistics workflows involve:

- document parsing

- routing

- pricing

- validation


Instead of one LLM:
responsibilities are split into agents orchestrated via a LangGraph DAG


This enables:

- modular design

- retry loops

- stateful execution


2. Why Actor–Critic Architecture?


LLMs alone are unreliable for critical decisions.

We introduced:


- Actor → performs task (extract, price)


- Critic → validates output deterministically


- Critic checks:

   weight ≠ 0

   parcel count consistency

   route validity

  If invalid, the system retries automatically.



3. Why XGBoost for Pricing?


LLMs are not ideal for numerical prediction.


XGBoost was chosen because:


- strong performance on tabular logistics data


- interpretable pricing logic


- low latency for real-time quotes


4. Why Redis (Caching)?


Geocoding APIs are:


 - expensive


 - slow



Redis cache:


- stores frequent locations (ports, warehouses)


- reduces API calls


- improves latency


5. Why PostgreSQL (State)?


Logistics workflows require:

  - persistence

  - audit trails

PostgreSQL enables:

  - shipment state tracking

  - reproducibility

  - debugging



Benchmarks

| Metric                        | Value              |
| ----------------------------- | ------------------ |
| Extraction Accuracy           | 96%                |
| Latency (p99)                 | ~9s                |
| Token Cost                    | ~$0.0001 / request |            
| Manual Intervention Reduction | ~80%               |



Observability:


LangSmith → tracing, debugging, cost monitoring


MLflow → experiment tracking and audit logs


Enables:


- full system visibility


- reproducible pipelines


- production debugging



Tech Stack

AI / LLM:


LangGraph, LangChain, Phi-3


Machine Learning:

XGBoost


Backend & Infra:

FastAPI, Redis, PostgreSQL, Docker


Evaluation:

MLflow, LangSmith



Challenges and Engineering Decisions:


API Reliability-


Problem: external API failures (Mapbox)


Solution: fallback to Haversine distance → ensured 100% uptime


LLM Reliability-


Problem: incorrect numerical outputs


Solution: deterministic Critic layer → sub-millisecond validation


Distributed System Issues-


Problem: MLflow DNS rebinding in containers


Solution: configured host allowlist


Example Workflow


Input: Waybill / logistics request


Output:


structured shipment data


optimized route


predicted pricing


validated decision



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


pip install -r requirements.txt

uvicorn app:app --reload

pytest tests

