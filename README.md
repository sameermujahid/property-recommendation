# AI-Powered Property Recommendation System

A production-grade, high-performance **Property Recommendation Platform** built using **Flask, PostgreSQL, SentenceTransformers, ChromaDB, and Parallel Processing**.

This system fetches real estate properties from a **PostgreSQL database**, stores embeddings inside a **persistent ChromaDB vector database**, and delivers intelligent, AI-powered recommendations using semantic search combined with weighted scoring algorithms.

Designed for **multi-user environments** and deployable on platforms like **Hugging Face Spaces**.

---

# Core Capabilities

## 1. AI Semantic Search Engine

* Uses `all-MiniLM-L6-v2` from SentenceTransformers
* Cosine similarity scoring
* Context-aware property matching
* Batch embedding generation for performance
* Hybrid recommendation logic (semantic + rule-based scoring)

---

## 2. PostgreSQL Data Layer

* Properties stored in PostgreSQL
* JSON image aggregation via SQL
* Feature parsing (JSON or comma-separated)
* Filter support:

  * Property type
  * Single or multiple price ranges
* Uses `psycopg2` with `RealDictCursor`

---

## 3. Vector Database (ChromaDB)

* Persistent storage at `property_db/`
* Cosine similarity index (HNSW)
* Metadata stored alongside embeddings
* Automatic cache refresh every 24 hours
* Background refresh (non-blocking)

---

## 4. Multi-User Architecture

Designed for concurrent users:

* Thread-safe request tracking
* Concurrent request limit protection
* Connection pooling for external API calls
* Parallel property processing (ThreadPoolExecutor)
* Background property fetching
* Non-blocking refresh logic
* Health monitoring endpoint

---

# Recommendation Scoring Logic

Each property is scored using weighted factors:

| Factor                   | Weight    |
| ------------------------ | --------- |
| Semantic similarity      | 50%       |
| Property type match      | 30%       |
| Price match              | 30%       |
| Feature detection        | up to 10% |
| Size bonus (>1000 sq ft) | 5%        |
| Room bonus (≥3 rooms)    | 5%        |

Scoring is performed in parallel for high performance.

---


# System Architecture

User → Flask API →
PostgreSQL (properties) →
ChromaDB (embeddings + metadata) →
Parallel scoring engine →
Top ranked properties returned →
Optional email delivery via SMTP

---

# Project Structure

```
├── app.py
├── property_db/              # Persistent ChromaDB storage
├── templates/
│   └── index1.html           # Full UI (multi-step form + browse)
├── requirements.txt
└── README.md
```

---

# API Endpoints

## Core Endpoints

| Method | Endpoint                      | Description                          |
| ------ | ----------------------------- | ------------------------------------ |
| GET    | `/`                           | Home page                            |
| GET    | `/all_properties`             | Returns all cached properties        |
| POST   | `/get_recommendations`        | Returns AI-ranked recommendations    |
| POST   | `/send_recommendations_email` | Sends recommendations via Gmail SMTP |
| GET    | `/search`                     | Full search                          |
| GET    | `/search_suggestions`         | Live suggestions                     |
| POST   | `/refresh_properties`         | Trigger background refresh           |
| GET    | `/cache_status`               | Cache diagnostics                    |
| GET    | `/health`                     | System health check                  |

---

# Performance Optimizations

* Parallel embedding generation
* Parallel property filtering
* Parallel scoring
* Background cache refresh
* Persistent vector DB
* Connection pooling
* Concurrent request protection
* CPU-aware worker scaling
* Hugging Face Spaces compatible (port 7860)

---

# Cache Strategy

* Properties cached in ChromaDB
* 24-hour expiry window
* Background refresh when stale
* Initial auto-fetch if cache empty
* Countdown scheduler thread
* Non-blocking refresh execution

---

# Email System

Two supported methods:

1. External Email API (via `EMAIL_API_URL`)
2. Gmail SMTP using `smtplib`

Supports:

* HTML formatted property cards
* Top N property selection
* Graceful error handling
* Timeout handling
* Connection failure detection

---

## Local

```
python app.py
```

Runs on:

```
http://localhost:7860
```
