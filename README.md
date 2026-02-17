# AI Powered Property Recommendation System

An intelligent, high-performance **Property Recommendation Engine** built using **Flask, SentenceTransformers, ChromaDB, and Parallel Processing**.

This system fetches real estate properties from an external API, stores them in a **vector database (ChromaDB)**, and delivers **AI-powered personalized recommendations** using semantic search + scoring algorithms.

---

## 🚀 Features

### 🔥 AI-Powered Semantic Search

* Uses `all-MiniLM-L6-v2` from SentenceTransformers
* Semantic similarity scoring
* Context-aware property matching

### ⚡ High-Performance Architecture

* Parallel batch processing (ThreadPoolExecutor)
* Multi-worker embedding generation
* Optimized property scoring
* Background cache refresh
* Connection pooling for API calls

### 🧠 Vector Database

* ChromaDB with cosine similarity
* Persistent storage
* Automatic cache refresh (24-hour expiry)

### 🏗 Multi-User Support

* Concurrent request handling
* Request tracking & rate limiting
* Background property fetching
* Thread-safe operations

### 📧 Account & Recommendation Flow

* Multi-step property preference form
* Email template system
* WhatsApp integration (Twilio ready)
* Guest mode supported

---

## 🛠 Tech Stack

### Backend

* Flask 
* Flask-CORS 
* SentenceTransformers 
* FAISS 
* ChromaDB 
* Scikit-learn 
* Pandas 

### Frontend

* Bootstrap 5 
* Multi-step form UI
* Dynamic property rendering

### Email Template

* HTML templating for property recommendations 

---

## 📂 Project Structure

```
├── app.py                  # Main Flask backend :contentReference[oaicite:9]{index=9}
├── requirements.txt        # Dependencies :contentReference[oaicite:10]{index=10}
├── templates/
│   ├── index.html          # Multi-step property form :contentReference[oaicite:11]{index=11}
│   └── email_template.html # Email recommendation template :contentReference[oaicite:12]{index=12}
├── property_db/            # ChromaDB persistent storage
└── README.md
```

---

## 🧠 How It Works

### 1️⃣ Property Fetching

* Fetches properties in parallel batches
* Uses connection pooling
* Background refresh support
* Automatic cache timestamp tracking

### 2️⃣ Vector Storage

* Property description → embedding
* Stored in ChromaDB
* Metadata stored alongside embeddings

### 3️⃣ Recommendation Engine

Each property is scored using:

* 🔎 Semantic similarity (50%)
* 🏠 Property type match (30%)
* 💰 Price range match (30%)
* 🌟 Feature detection
* 📐 Size & room bonus scoring

Parallel scoring ensures fast response even with large datasets.

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/sameermujahid/property-recommendation
cd property-recommendation
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies are defined in:


---

## ▶️ Run the Application

```bash
python app.py
```

Server runs on:

```
http://localhost:5000
```

---

## 📊 Performance Optimizations

* ✅ Parallel property fetching
* ✅ Parallel embedding generation
* ✅ Parallel scoring
* ✅ Background cache refresh
* ✅ Multi-thread safe architecture
* ✅ Connection pooling
* ✅ Batch processing

---

## 🗄 Cache Strategy

* Properties cached for **24 hours**
* Automatic background refresh
* Non-blocking updates
* Persistent vector storage

---

## 🔐 Environment Variables (Recommended)

Instead of hardcoding:

```
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
BACKEND_API_URL
```

Use `.env` file for production security.

---

## 📦 API Endpoints (Example)

| Method | Endpoint          | Description                  |
| ------ | ----------------- | ---------------------------- |
| GET    | `/`               | Home page                    |
| POST   | `/recommend`      | Get property recommendations |
| POST   | `/create-account` | Create user account          |

---

## 📈 Future Improvements

* Redis caching layer
* Async FastAPI migration
* Docker deployment
* Kubernetes scaling
* User history personalization
* ML-based re-ranking
* Production logging system

---
