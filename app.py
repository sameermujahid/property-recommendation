from flask import Flask, render_template, request, jsonify
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import urllib3
import chromadb
from chromadb.config import Settings
import time
from flask_cors import CORS
import logging
import threading
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import psutil
import psycopg2
from psycopg2.extras import RealDictCursor
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from flask import request, jsonify

TARGET_PROPERTIES = 20  # Maximum number of properties to return

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable SSL warning since we're using https://localhost
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)

# Multi-user support configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Configure CORS with specific settings for multi-user support
CORS(app, 
     resources={r"/*": {
         "origins": ["http://localhost:4200", "https://huggingface.co", "*"],
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
         "supports_credentials": True,
         "max_age": 3600
     }})

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Initialize the sentence transformer model with error handling
import multiprocessing as mp
from functools import partial

# Load model directly to avoid multiprocessing issues
try:
    logger.info("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise
# PostgreSQL configuration
DB_CONFIG = "postgresql://neondb_owner:npg_76laiVosQFHx@ep-sparkling-voice-ai150vc2-pooler.c-4.us-east-1.aws.neon.tech/neondb?sslmode=require"

# Configuration
TOTAL_PROPERTIES = 500  # Total number of properties to fetch
CACHE_EXPIRY_SECONDS = 24 * 60 * 60  # 24 hours in seconds
BATCH_SIZE = 50  # Number of properties per batch
MAX_WORKERS = 10  # Number of parallel workers
# REQUEST_TIMEOUT = None  # No timeout for requests - removed

# Multi-user support configuration
MAX_CONCURRENT_REQUESTS = 50  # Maximum concurrent requests
REQUEST_RATE_LIMIT = 100  # Requests per minute per IP
CONNECTION_POOL_SIZE = 20  # Connection pool size for external API calls

# Create connection pool for multi-user support
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=CONNECTION_POOL_SIZE,
    pool_maxsize=CONNECTION_POOL_SIZE,
    max_retries=3
)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Global variable for property collection
property_collection = None

# Path for cache timestamp
CACHE_TIMESTAMP_PATH = os.path.join("property_db", "last_cache_time.txt")

# Global variable for countdown thread
countdown_thread = None
stop_countdown = False
countdown_initialized = False  # Add flag to track initialization

# Multi-user request tracking
active_requests = 0
request_lock = threading.Lock()

# Background task management
background_tasks = {}
task_lock = threading.Lock()
is_fetching_properties = False
fetch_lock = threading.Lock()
def _parse_features(key_features):
    """Parse key_features from DB - supports JSON array or plain text"""
    if not key_features:
        return []
    try:
        parsed = json.loads(key_features)
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except (json.JSONDecodeError, TypeError):
        return [f.strip() for f in str(key_features).split(',') if f.strip()]
def _row_to_property_dict(row, images):
    """Convert PostgreSQL row to the format expected by the app (camelCase)"""
    return {
        'id': str(row['property_id']),
        'propertyName': row['property_name'] or '',
        'typeName': row['property_type'] or '',
        'description': row['description'] or '',
        'address': row['address'] or '',
        'totalSquareFeet': float(row['total_square_feet'] or 0),
        'beds': int(row['beds'] or 0),
        'baths': int(row['baths'] or 0),
        'numberOfRooms': int(row['number_of_rooms'] or 0),
        'marketValue': float(row['market_value'] or 0),
        'yearBuilt': str(row['year_built'] or ''),
        'propertyImages': [{'imageUrl': img['image_url']} for img in images],
        'features': _parse_features(row.get('key_features')),
        'location': {
            'city': row.get('city'),
            'state': row.get('state'),
            'country': row.get('country'),
            'latitude': float(row['latitude']) if row.get('latitude') else None,
            'longitude': float(row['longitude']) if row.get('longitude') else None
        },
        'floorPlans': [],
        'documents': [],
        'propertyVideos': []
    }
def fetch_properties_from_db(property_type=None, min_price=None, max_price=None):
    """Fetch all properties from PostgreSQL (with optional filters)"""
    conn = None
    try:
        conn = psycopg2.connect(DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT p.*, 
                   COALESCE(
                       (SELECT json_agg(json_build_object('image_url', pi.image_url) ORDER BY pi.sort_order)
                        FROM property_images pi 
                        WHERE pi.property_id = p.property_id),
                       '[]'::json
                   ) AS images
            FROM properties p
            WHERE p.is_deleted = FALSE
        """
        params = []

        # Normalize min_price / max_price: take the first numeric value if lists are passed
        if isinstance(min_price, list):
            min_price = next((v for v in min_price if v is not None), None)
        if isinstance(max_price, list):
            max_price = next((v for v in max_price if v is not None), None)
        if property_type:
            query += " AND p.property_type ILIKE %s"
            params.append(f"%{property_type}%")
        if min_price is not None:
            query += " AND (p.market_value IS NULL OR p.market_value >= %s)"
            params.append(min_price)
        if max_price is not None:
            query += " AND (p.market_value IS NULL OR p.market_value <= %s)"
            params.append(max_price)
        
        query += " ORDER BY p.property_id"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        
        properties = []
        for row in rows:
            images = row['images'] if isinstance(row['images'], list) else (json.loads(row['images']) if row['images'] else [])
            prop = _row_to_property_dict(dict(row), images)
            properties.append(prop)
        
        return {"data": properties}
    except Exception as e:
        logger.error(f"Error fetching from PostgreSQL: {e}")
        return None
    finally:
        if conn:
            conn.close()
def format_time_remaining(seconds):
    """Format remaining time in a human-readable format"""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes} minutes {remaining_seconds} seconds"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours} hours {minutes} minutes"


def fetch_properties_background_safe(property_type=None, min_price=None, max_price=None):
    """Fetch properties from PostgreSQL database"""
    global is_fetching_properties

    with fetch_lock:
        if is_fetching_properties:
            logger.info("Property fetch already in progress, skipping...")
            return None
        is_fetching_properties = True

    try:
        logger.info("🔄 Starting background property fetch from PostgreSQL...")
        start_time = time.time()

        result = fetch_properties_from_db(property_type=property_type, min_price=min_price, max_price=max_price)

        end_time = time.time()
        fetch_duration = end_time - start_time

        if result and result.get('data'):
            all_properties = result['data']
            logger.info(f"✅ Successfully fetched {len(all_properties)} properties from PostgreSQL in {fetch_duration:.2f} seconds")
            return {"data": all_properties}
        else:
            logger.error(f"❌ Failed to fetch any properties after {fetch_duration:.2f} seconds")
            return None

    except Exception as e:
        logger.error(f"❌ Error during background property fetch: {str(e)}")
        return None
    finally:
        with fetch_lock:
            is_fetching_properties = False
def fetch_properties(property_type=None, min_price=None, max_price=None):
    """Fetch properties using parallel workers and batches - now calls background version"""
    return fetch_properties_background_safe(property_type, min_price, max_price)

def update_cache_timestamp():
    try:
        os.makedirs(os.path.dirname(CACHE_TIMESTAMP_PATH), exist_ok=True)
        current_time = time.time()
        with open(CACHE_TIMESTAMP_PATH, 'w') as f:
            f.write(str(current_time))
        next_update = datetime.fromtimestamp(current_time + CACHE_EXPIRY_SECONDS)
        logger.info(f"Updated cache timestamp. Next update scheduled for: {next_update.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Next update in: {format_time_remaining(CACHE_EXPIRY_SECONDS)}")
    except Exception as e:
        logger.error(f"Error updating cache timestamp: {e}")

def refresh_properties_background():
    """Refresh properties in background without blocking the main application"""
    try:
        logger.info("🔄 Starting background property refresh...")
        data = fetch_properties_background_safe()
        if data and 'data' in data:
            new_properties = [p for p in data['data'] if isinstance(p, dict)]
            if new_properties:
                # Use ThreadPoolExecutor for non-blocking cache update
                def update_cache_async():
                    try:
                        update_property_cache(new_properties)
                        update_cache_timestamp()
                        logger.info("✅ Background property refresh completed successfully")
                    except Exception as e:
                        logger.error(f"Error updating cache in background: {e}")
                
                # Start cache update in background thread
                threading.Thread(target=update_cache_async, daemon=True).start()
            else:
                logger.warning("⚠️ No properties fetched during background refresh")
        else:
            logger.warning("⚠️ Failed to fetch properties during background refresh")
    except Exception as e:
        logger.error(f"❌ Error during background property refresh: {str(e)}")

def force_initial_property_fetch():
    """Force initial property fetch to ensure cache is created - non-blocking"""
    try:
        logger.info("🔄 Force initial property fetch...")
        data = fetch_properties_background_safe()
        if data and 'data' in data:
            new_properties = [p for p in data['data'] if isinstance(p, dict)]
            if new_properties:
                # Use ThreadPoolExecutor for non-blocking cache update
                def update_cache_async():
                    try:
                        update_property_cache(new_properties)
                        update_cache_timestamp()
                        logger.info(f"✅ Force initial fetch completed with {len(new_properties)} properties")
                    except Exception as e:
                        logger.error(f"Error updating cache in force initial fetch: {e}")
                
                # Start cache update in background thread
                threading.Thread(target=update_cache_async, daemon=True).start()
                return True
            else:
                logger.warning("⚠️ No properties fetched during force initial fetch")
                return False
        else:
            logger.warning("⚠️ Failed to fetch properties during force initial fetch")
            return False
    except Exception as e:
        logger.error(f"❌ Error during force initial property fetch: {str(e)}")
        return False

def process_property_batch(properties_batch, existing_ids):
    """Process a batch of properties for ChromaDB in parallel"""
    batch_ids = []
    batch_metadatas = []
    batch_documents = []
    batch_embeddings = []
    
    # Prepare property texts for batch embedding
    property_texts = []
    valid_properties = []
    
    for prop in properties_batch:
        try:
            prop_id = str(prop.get('id'))
            if prop_id not in existing_ids:
                # Create property text for embedding
                property_text = f"{prop.get('propertyName', '')} {prop.get('typeName', '')} {prop.get('description', '')} {prop.get('address', '')}"
                property_texts.append(property_text)
                valid_properties.append(prop)
        except Exception as e:
            logger.error(f"Error processing property for batch: {str(e)}")
            continue
    
    if not valid_properties:
        return batch_ids, batch_metadatas, batch_documents, batch_embeddings
    
    # Create embeddings in batch for better performance
    try:
        embeddings = model.encode(property_texts)
        
        for i, prop in enumerate(valid_properties):
            try:
                prop_id = str(prop.get('id'))
                
                # Convert property to metadata dictionary with proper type handling
                metadata = {
                    'id': str(prop.get('id', '')),
                    'propertyName': str(prop.get('propertyName', '')),
                    'typeName': str(prop.get('typeName', '')),
                    'description': str(prop.get('description', '')),
                    'address': str(prop.get('address', '')),
                    'totalSquareFeet': float(prop.get('totalSquareFeet', 0)),
                    'beds': int(prop.get('beds', 0)),
                    'baths': int(prop.get('baths', 0)),
                    'numberOfRooms': int(prop.get('numberOfRooms', 0)),
                    'marketValue': float(prop.get('marketValue', 0)),
                    'yearBuilt': str(prop.get('yearBuilt', '')),
                    'propertyImages': json.dumps(prop.get('propertyImages', [])),
                    'features': json.dumps(prop.get('features', [])),
                    'location': json.dumps(prop.get('location', {})),
                    'floorPlans': json.dumps(prop.get('floorPlans', [])),
                    'documents': json.dumps(prop.get('documents', [])),
                    'propertyVideos': json.dumps(prop.get('propertyVideos', []))
                }
                
                batch_ids.append(prop_id)
                batch_metadatas.append(metadata)
                batch_documents.append(property_texts[i])
                batch_embeddings.append(embeddings[i].tolist())
            except Exception as e:
                logger.error(f"Error processing property {i} in batch: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"Error creating embeddings for batch: {str(e)}")
    
    return batch_ids, batch_metadatas, batch_documents, batch_embeddings

def update_property_cache(new_properties):
    """Update property cache with parallel processing optimization"""
    global property_collection
    try:
        if not new_properties:
            logger.warning("No new properties to add")
            return

        logger.info(f"Updating ChromaDB with {len(new_properties)} properties using parallel processing")
        start_time = time.time()
        
        # Get existing property IDs
        existing_results = property_collection.get()
        existing_ids = set(existing_results['ids']) if existing_results and existing_results['ids'] else set()
        
        # Filter out existing properties in parallel
        def filter_new_property(prop):
            try:
                prop_id = str(prop.get('id'))
                if prop_id not in existing_ids:
                    return prop
                return None
            except Exception as e:
                logger.error(f"Error filtering property: {str(e)}")
                return None
        
        # Use ThreadPoolExecutor for parallel filtering
        max_workers = min(mp.cpu_count(), 8)
        new_properties_filtered = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_property = {
                executor.submit(filter_new_property, prop): i 
                for i, prop in enumerate(new_properties)
            }
            
            for future in as_completed(future_to_property):
                try:
                    filtered_prop = future.result()
                    if filtered_prop:
                        new_properties_filtered.append(filtered_prop)
                except Exception as e:
                    property_idx = future_to_property[future]
                    logger.error(f"Error filtering property {property_idx}: {str(e)}")
        
        if not new_properties_filtered:
            logger.info("No new properties to add to ChromaDB")
            return
        
        logger.info(f"Processing {len(new_properties_filtered)} new properties in parallel")
        
        # Process properties in batches for better performance
        cache_batch_size = 100  # Process 100 properties at a time for embedding
        all_ids = []
        all_metadatas = []
        all_documents = []
        all_embeddings = []
        
        # Process in batches with parallel processing
        def process_batch_parallel(batch):
            return process_property_batch(batch, existing_ids)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create batches
            batches = [new_properties_filtered[i:i + cache_batch_size] 
                      for i in range(0, len(new_properties_filtered), cache_batch_size)]
            
            # Submit batch processing tasks
            future_to_batch = {
                executor.submit(process_batch_parallel, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_ids, batch_metadatas, batch_documents, batch_embeddings = future.result()
                    all_ids.extend(batch_ids)
                    all_metadatas.extend(batch_metadatas)
                    all_documents.extend(batch_documents)
                    all_embeddings.extend(batch_embeddings)
                    logger.info(f"Processed batch {batch_idx + 1}/{len(batches)} - {len(batch_ids)} properties")
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
        
        # Add all new properties to ChromaDB
        if all_ids:
            logger.info(f"Adding {len(all_ids)} new properties to ChromaDB")
            property_collection.add(
                ids=all_ids,
                metadatas=all_metadatas,
                documents=all_documents,
                embeddings=all_embeddings
            )
            end_time = time.time()
            logger.info(f"✅ Successfully added {len(all_ids)} properties to ChromaDB in {end_time - start_time:.2f} seconds using parallel processing")
        else:
            logger.info("No new properties to add to ChromaDB")
            
    except Exception as e:
        logger.error(f"Error updating property cache: {str(e)}")
        raise

def is_cache_stale():
    """Check if cache is stale and needs refresh"""
    if not os.path.exists(CACHE_TIMESTAMP_PATH):
        logger.info("No cache timestamp found, cache needs refresh")
        return True
    try:
        with open(CACHE_TIMESTAMP_PATH, 'r') as f:
            last_time = float(f.read().strip())
        current_time = time.time()
        time_diff = current_time - last_time
        is_stale = time_diff > CACHE_EXPIRY_SECONDS
        
        if is_stale:
            logger.info(f"Cache is stale (older than {format_time_remaining(CACHE_EXPIRY_SECONDS)}). Last update: {datetime.fromtimestamp(last_time)}")
            logger.info(f"Time since last update: {format_time_remaining(time_diff)}")
        else:
            time_remaining = CACHE_EXPIRY_SECONDS - time_diff
            next_update = datetime.fromtimestamp(current_time + time_remaining)
            logger.info(f"Cache is fresh. Last update: {datetime.fromtimestamp(last_time)}")
            logger.info(f"Next update in: {format_time_remaining(time_remaining)}")
            logger.info(f"Next update scheduled for: {next_update.strftime('%Y-%m-%d %H:%M:%S')}")
        return is_stale
    except Exception as e:
        logger.error(f"Error reading cache timestamp: {e}, cache needs refresh")
        return True

def initialize_collection():
    global property_collection
    try:
        # First try to get the collection
        try:
            property_collection = chroma_client.get_collection("properties")
            logger.info("Using existing property collection from vector DB")
        except Exception as e:
            logger.info("Creating new property collection")
            property_collection = chroma_client.create_collection(
                name="properties",
                metadata={"hnsw:space": "cosine"}
            )
        
        # Check if collection is empty or needs refresh
        results = property_collection.get()
        if not results or not results['ids']:
            logger.info("Collection is empty, fetching initial properties")
            # Always fetch initial properties if collection is empty
            data = fetch_properties()
            if data and 'data' in data:
                new_properties = [p for p in data['data'] if isinstance(p, dict)]
                if new_properties:
                    update_property_cache(new_properties)
                    update_cache_timestamp()
                    logger.info(f"✅ Successfully added {len(new_properties)} initial properties to collection")
                else:
                    logger.warning("No properties fetched from API during initialization")
            else:
                logger.warning("Failed to fetch properties from API during initialization")
        else:
            logger.info(f"Collection has {len(results['ids'])} properties, checking if cache is fresh")
            if is_cache_stale():
                logger.info("Cache is stale, will refresh in background")
                # Start background refresh
                threading.Thread(target=refresh_properties_background, daemon=True).start()
            else:
                logger.info("Cache is fresh, using existing properties")
    except Exception as e:
        logger.error(f"Error initializing collection: {str(e)}")
        # Try to recreate the collection
        try:
            logger.info("Attempting to recreate collection")
            try:
                chroma_client.delete_collection("properties")
            except:
                pass
            property_collection = chroma_client.create_collection(
                name="properties",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Successfully recreated collection")
        except Exception as recreate_error:
            logger.error(f"Error recreating collection: {str(recreate_error)}")
            raise

# Initialize ChromaDB with persistent storage
try:
    logger.info("Initializing ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="property_db")
    # Initialize collection at startup
    initialize_collection()
    logger.info("ChromaDB initialized successfully")
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {str(e)}")
    raise

def calculate_property_score(prop, price_range=None, property_type=None, user_input=None):
    """Calculate property score with optimized parallel processing"""
    score = 0.0
    similarity_score = 0.0
    
    # Calculate semantic similarity if user input is provided
    if user_input:
        property_text = f"{prop.get('propertyName', '')} {prop.get('typeName', '')} {prop.get('description', '')} {prop.get('address', '')}"
        property_embedding = model.encode([property_text])[0]
        user_embedding = model.encode([user_input])[0]
        similarity_score = float(cosine_similarity([property_embedding], [user_embedding])[0][0])  # Convert to Python float
        score += similarity_score * 0.5  # 50% weight for semantic similarity
    
    # Base score for property type match (only if property_type is specified)
    if property_type:
        prop_type = prop.get('typeName', '').lower()
        if isinstance(property_type, list):
            # Check if property type matches any in the list
            if any(pt.lower() == prop_type for pt in property_type):
                score += 0.3  # 30% weight for exact type match
            elif any(pt.lower() in prop_type for pt in property_type):
                score += 0.2  # 20% weight for partial type match
        else:
            # Single property type
            if prop_type == property_type.lower():
                score += 0.3  # 30% weight for exact type match
            elif property_type.lower() in prop_type:
                score += 0.2  # 20% weight for partial type match
    else:
        # For additional properties (no type filter), give base score
        score += 0.1  # Base score for any property type
    
    # Score for price range match
    if price_range:
        price = float(prop.get('marketValue', 0))  # Convert to Python float
        if isinstance(price_range, list) and len(price_range) > 0:
            if isinstance(price_range[0], list):
                # Multiple price ranges - check if price falls in any range
                price_in_range = False
                for range_item in price_range:
                    min_val = range_item[0] if range_item[0] is not None else 0
                    max_val = range_item[1] if range_item[1] is not None else float('inf')
                    if min_val <= price <= max_val:
                        price_in_range = True
                        score += 0.3  # 30% weight for exact price match
                        break
                if not price_in_range:
                    # Check if price is close to any range
                    for range_item in price_range:
                        min_val = range_item[0] if range_item[0] is not None else 0
                        max_val = range_item[1] if range_item[1] is not None else float('inf')
                        if price < min_val:
                            price_diff = (min_val - price) / min_val if min_val > 0 else 1
                            score += 0.1 * (1 - min(price_diff, 1))  # Up to 10% for properties below range
                        elif price < max_val * 1.2:  # Allow 20% above max price
                            price_diff = (price - max_val) / max_val if max_val > 0 else 1
                            score += 0.2 * (1 - min(price_diff, 1))  # Up to 20% for properties above range
            else:
                # Single price range
                min_val = price_range[0] if price_range[0] is not None else 0
                max_val = price_range[1] if price_range[1] is not None else float('inf')
                if min_val <= price <= max_val:
                    score += 0.3  # 30% weight for exact price match
                elif price < min_val:
                    # Properties below range get a small score
                    price_diff = (min_val - price) / min_val if min_val > 0 else 1
                    score += 0.1 * (1 - min(price_diff, 1))  # Up to 10% for properties below range
                elif price < max_val * 1.2:  # Allow 20% above max price
                    price_diff = (price - max_val) / max_val if max_val > 0 else 1
                    score += 0.2 * (1 - min(price_diff, 1))  # Up to 20% for properties above range
    
    # Score for property features
    features = [
        'balcony', 'parking', 'security', 'garden',
        'swimming pool', 'gym', 'power backup', 'water supply'
    ]
    description = prop.get('description', '')
    if description:  # Only process features if description exists
        description = description.lower()
        feature_count = sum(1 for feature in features if feature in description)
        score += (feature_count / len(features)) * 0.1  # Up to 10% for features
    
    # Score for property size
    if prop.get('totalSquareFeet', 0) > 1000:
        score += 0.05  # 5% for larger properties
    
    # Score for number of rooms
    if prop.get('numberOfRooms', 0) >= 3:
        score += 0.05  # 5% for properties with more rooms
    
    return float(score), float(similarity_score)  # Convert to Python floats

def calculate_property_scores_parallel(properties, price_range=None, property_type=None, user_input=None, max_workers=None):
    """Calculate scores for multiple properties in parallel"""
    if not properties:
        return []
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # Use CPU cores but cap at 8
    
    logger.info(f"Calculating scores for {len(properties)} properties using {max_workers} workers")
    
    # Prepare property texts for batch embedding if user_input is provided
    if user_input:
        property_texts = []
        for prop in properties:
            property_text = f"{prop.get('propertyName', '')} {prop.get('typeName', '')} {prop.get('description', '')} {prop.get('address', '')}"
            property_texts.append(property_text)
        
        # Batch encode all property texts at once
        try:
            property_embeddings = model.encode(property_texts)
            user_embedding = model.encode([user_input])[0]
            
            # Calculate similarities in batch
            similarities = cosine_similarity(property_embeddings, [user_embedding]).flatten()
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            similarities = [0.0] * len(properties)
    else:
        similarities = [0.0] * len(properties)
    
    # Process properties in parallel
    def score_single_property(args):
        prop, similarity = args
        score = 0.0
        
        # Add similarity score
        score += similarity * 0.5
        
        # Base score for property type match
        if property_type:
            prop_type = prop.get('typeName', '').lower()
            if isinstance(property_type, list):
                if any(pt.lower() == prop_type for pt in property_type):
                    score += 0.3
                elif any(pt.lower() in prop_type for pt in property_type):
                    score += 0.2
            else:
                if prop_type == property_type.lower():
                    score += 0.3
                elif property_type.lower() in prop_type:
                    score += 0.2
        else:
            score += 0.1
        
        # Price range scoring
        if price_range:
            price = float(prop.get('marketValue', 0))
            if isinstance(price_range, list) and len(price_range) > 0:
                if isinstance(price_range[0], list):
                    price_in_range = False
                    for range_item in price_range:
                        min_val = range_item[0] if range_item[0] is not None else 0
                        max_val = range_item[1] if range_item[1] is not None else float('inf')
                        if min_val <= price <= max_val:
                            price_in_range = True
                            score += 0.3
                            break
                    if not price_in_range:
                        for range_item in price_range:
                            min_val = range_item[0] if range_item[0] is not None else 0
                            max_val = range_item[1] if range_item[1] is not None else float('inf')
                            if price < min_val:
                                price_diff = (min_val - price) / min_val if min_val > 0 else 1
                                score += 0.1 * (1 - min(price_diff, 1))
                            elif price < max_val * 1.2:  # Allow 20% above max price
                                price_diff = (price - max_val) / max_val if max_val > 0 else 1
                                score += 0.2 * (1 - min(price_diff, 1))
                else:
                    min_val = price_range[0] if price_range[0] is not None else 0
                    max_val = price_range[1] if price_range[1] is not None else float('inf')
                    if min_val <= price <= max_val:
                        score += 0.3
                    elif price < min_val:
                        price_diff = (min_val - price) / min_val if min_val > 0 else 1
                        score += 0.1 * (1 - min(price_diff, 1))
                    elif price < max_val * 1.2:  # Allow 20% above max price
                        price_diff = (price - max_val) / max_val if max_val > 0 else 1
                        score += 0.2 * (1 - min(price_diff, 1))
        
        # Feature scoring
        features = ['balcony', 'parking', 'security', 'garden', 'swimming pool', 'gym', 'power backup', 'water supply']
        description = prop.get('description', '')
        if description:
            description = description.lower()
            feature_count = sum(1 for feature in features if feature in description)
            score += (feature_count / len(features)) * 0.1
        
        # Size and room scoring
        if prop.get('totalSquareFeet', 0) > 1000:
            score += 0.05
        if prop.get('numberOfRooms', 0) >= 3:
            score += 0.05
        
        return {
            'property': prop,
            'score': float(score),
            'similarity': float(similarity)
        }
    
    # Use ThreadPoolExecutor for parallel scoring
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create arguments for each property
        args = list(zip(properties, similarities))
        
        # Submit all scoring tasks
        future_to_property = {
            executor.submit(score_single_property, arg): i 
            for i, arg in enumerate(args)
        }
        
        # Collect results
        scored_properties = []
        for future in as_completed(future_to_property):
            try:
                result = future.result()
                if result['score'] >= 0.1:  # Filter low scores
                    scored_properties.append(result)
            except Exception as e:
                property_idx = future_to_property[future]
                logger.error(f"Error scoring property {property_idx}: {e}")
    
    # Sort by score
    scored_properties.sort(key=lambda x: x['score'], reverse=True)
    
    # Convert back to original format
    for result in scored_properties:
        result['property']['score'] = result['score']
        result['property']['similarity'] = result['similarity']
    
    logger.info(f"Successfully scored {len(scored_properties)} properties in parallel")
    return [result['property'] for result in scored_properties]

def get_recommendations(user_input, price_range=None, property_type=None):
    global property_collection
    
    # Handle multiple price ranges - convert to min/max arrays
    min_price = None
    max_price = None
    if price_range:
        if isinstance(price_range, list) and len(price_range) > 0:
            if isinstance(price_range[0], list):
                # Array of price ranges
                min_price = [range[0] for range in price_range]
                max_price = [range[1] for range in price_range]
            else:
                # Single price range
                min_price = [price_range[0]]
                max_price = [price_range[1]]
    
    # Get properties from ChromaDB, fetch from API if empty
    try:
        logger.info("Getting properties from ChromaDB...")
        properties = get_cached_properties(
            property_type=property_type,
            min_price=min_price,
            max_price=max_price
        )
        
        if not properties:
            logger.warning("No properties found in ChromaDB, fetching from API...")
            # Fetch properties from API and store in ChromaDB
            data = fetch_properties(
                property_type=property_type,
                min_price=min_price[0] if min_price else None,
                max_price=max_price[0] if max_price else None
            )
            if data and 'data' in data:
                new_properties = [p for p in data['data'] if isinstance(p, dict)]
                if new_properties:
                    # Store properties in ChromaDB
                    update_property_cache(new_properties)
                    update_cache_timestamp()
                    logger.info(f"✅ Fetched and stored {len(new_properties)} properties from API")
                    
                    # Now get properties from ChromaDB again
                    properties = get_cached_properties(
                        property_type=property_type,
                        min_price=min_price,
                        max_price=max_price
                    )
                else:
                    logger.warning("No properties fetched from API")
                    return []
            else:
                logger.warning("Failed to fetch properties from API")
                return []
    except Exception as e:
        logger.error(f"Error getting properties from ChromaDB: {str(e)}")
        properties = []
    
    # Return empty list if no properties found matching criteria
    if not properties:
        logger.info("No properties found matching criteria, returning empty list")
        return []
    
    # Calculate scores for properties using parallel processing
    scored_properties = calculate_property_scores_parallel(
        properties, price_range, property_type, user_input, max_workers=min(mp.cpu_count(), 8)
    )
    
    # If we have enough properties matching exact criteria, return them
    if len(scored_properties) >= TARGET_PROPERTIES:
        logger.info(f"Found {len(scored_properties)} properties matching exact criteria, returning top {TARGET_PROPERTIES}")
        return scored_properties[:TARGET_PROPERTIES]
    
    # Return only exact matches, no additional properties
    logger.info(f"Found {len(scored_properties)} properties matching exact criteria, returning all matches")
    return scored_properties

def get_cached_properties(property_type=None, min_price=None, max_price=None):
    """Get properties from ChromaDB with parallel processing optimization"""
    global property_collection
    try:
        # Verify collection exists and has data
        if not property_collection:
            logger.info("Collection not initialized, reinitializing...")
            initialize_collection()
            if not property_collection:
                logger.error("Failed to initialize collection")
                return []
        
        logger.info("Fetching properties from ChromaDB with parallel processing...")
        # Get all properties from ChromaDB
        results = property_collection.get()
        if not results or not results['ids']:
            logger.warning("No properties found in ChromaDB")
            return []
        
        logger.info(f"Found {len(results['ids'])} properties in ChromaDB")
        
        # Process ChromaDB results in parallel
        def process_single_property(args):
            i, metadata = args
            try:
                # Convert JSON strings back to Python objects and handle type conversion
                property_data = {
                    'id': metadata['id'],
                    'propertyName': metadata['propertyName'],
                    'typeName': metadata['typeName'],
                    'description': metadata['description'],
                    'address': metadata['address'],
                    'totalSquareFeet': float(metadata['totalSquareFeet']),
                    'beds': int(metadata['beds']),
                    'baths': int(metadata['baths']),
                    'numberOfRooms': int(metadata['numberOfRooms']),
                    'marketValue': float(metadata['marketValue']),
                    'yearBuilt': metadata['yearBuilt'],
                    'propertyImages': json.loads(metadata['propertyImages']),
                    'features': json.loads(metadata['features']),
                    'location': json.loads(metadata['location']),
                    'floorPlans': json.loads(metadata['floorPlans']),
                    'documents': json.loads(metadata['documents']),
                    'propertyVideos': json.loads(metadata['propertyVideos'])
                }
                return property_data
            except Exception as e:
                logger.error(f"Error processing property {i}: {str(e)}")
                return None
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(mp.cpu_count(), 8)  # Use CPU cores but cap at 8
        properties = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create arguments for each property
            args = [(i, results['metadatas'][i]) for i in range(len(results['ids']))]
            
            # Submit all property processing tasks
            future_to_index = {
                executor.submit(process_single_property, arg): i 
                for i, arg in enumerate(args)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                try:
                    property_data = future.result()
                    if property_data:
                        properties.append(property_data)
                except Exception as e:
                    index = future_to_index[future]
                    logger.error(f"Error processing property {index}: {str(e)}")
        
        logger.info(f"Successfully processed {len(properties)} properties from ChromaDB")
        
        # Apply filters in parallel
        def filter_single_property(prop):
            try:
                # Apply property type filter (handle both single string and array)
                if property_type:
                    prop_type = prop.get('typeName', '').lower()
                    if isinstance(property_type, list):
                        # Check if property type matches any in the list
                        if not any(pt.lower() == prop_type for pt in property_type):
                            return None
                    else:
                        # Single property type
                        if prop_type != property_type.lower():
                            return None
                    
                # Apply price range filter (handle both single range and array of ranges)
                if min_price is not None and max_price is not None:
                    price = prop.get('marketValue', 0)
                    if isinstance(min_price, list) and isinstance(max_price, list):
                        # Multiple price ranges - check if price falls in any range
                        price_in_range = False
                        for i in range(len(min_price)):
                            min_val = min_price[i] if min_price[i] is not None else 0
                            max_val = max_price[i] if max_price[i] is not None else float('inf')
                            if min_val <= price <= max_val:
                                price_in_range = True
                                break
                        if not price_in_range:
                            return None
                    else:
                        # Single price range
                        min_val = min_price if min_price is not None else 0
                        max_val = max_price if max_price is not None else float('inf')
                        if price < min_val or price > max_val:
                            return None
                
                return prop
            except Exception as e:
                logger.error(f"Error filtering property: {str(e)}")
                return None
        
        # Filter properties in parallel
        filtered_properties = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all filtering tasks
            future_to_property = {
                executor.submit(filter_single_property, prop): i 
                for i, prop in enumerate(properties)
            }
            
            # Collect filtered results
            for future in as_completed(future_to_property):
                try:
                    filtered_prop = future.result()
                    if filtered_prop:
                        filtered_properties.append(filtered_prop)
                except Exception as e:
                    property_idx = future_to_property[future]
                    logger.error(f"Error filtering property {property_idx}: {str(e)}")
        
        logger.info(f"Filtered to {len(filtered_properties)} properties from ChromaDB using parallel processing")
        return filtered_properties
    except Exception as e:
        logger.error(f"Error getting properties from ChromaDB: {str(e)}")
        return []
@app.route('/')
def home():
    try:
        properties = get_cached_properties()

        # ================================
        # 🔹 Extract unique property types
        # ================================
        property_types = sorted(
            list({p['typeName'] for p in properties if p.get('typeName')})
        )

        # ================================
        # 🔹 Generate CLEAN price ranges
        # ================================
        prices = [p['marketValue'] for p in properties if p.get('marketValue')]
        price_ranges = []

        if prices:
            min_price = int(min(prices))
            max_price = int(max(prices))

            # Round nicely (nearest 10 lakh)
            rounded_min = math.floor(min_price / 1_000_000) * 1_000_000
            rounded_max = math.ceil(max_price / 1_000_000) * 1_000_000

            if rounded_max <= rounded_min:
                rounded_max = rounded_min + 1_000_000

            # Create 4 clean buckets
            step = (rounded_max - rounded_min) // 4

            current = rounded_min

            while current < rounded_max:
                upper = current + step

                # Round each range nicely (nearest 1 lakh)
                start_clean = round(current, -5)
                end_clean = round(upper, -5)

                price_ranges.append({
                    "label": f"₹{start_clean:,} - ₹{end_clean:,}",
                    "min": start_clean,
                    "max": end_clean
                })

                current = upper

        # ================================
        # 🔹 Extract unique amenities
        # ================================
        amenities = set()
        for p in properties:
            for f in p.get('features', []):
                amenities.add(f)

        amenities = sorted(list(amenities))

        return render_template(
            'index1.html',
            property_types=property_types,
            price_ranges=price_ranges,
            amenities=amenities
        )

    except Exception as e:
        logger.error(f"Error loading home page: {str(e)}")
        return render_template(
            'index1.html',
            property_types=[],
            price_ranges=[],
            amenities=[]
        )
import base64
import logging
from flask import request, jsonify
import requests

TWILIO_ACCOUNT_SID = "AC3bb0a3ea71812794b85b75ccf7d6ad07"
TWILIO_AUTH_TOKEN = "ddba202f5f4a48d08f04fb983a161fa3"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"
WHATSAPP_CHAR_LIMIT = 1600
PROPERTY_BASE_URL = "url"
MAX_PROPERTIES_TO_SEND = 3  # Send only 1 property in email
EMAIL_API_URL = "url"
def send_email_recommendations(email, first_name, last_name, recommendations):
    try:
        # Take only the top properties
        top_recommendations = recommendations[:MAX_PROPERTIES_TO_SEND]

        # Prepare email content
        subject = f"Your Top {MAX_PROPERTIES_TO_SEND} Property Recommendation"

        # Generate properties HTML
        properties_html = ""
        for idx, prop in enumerate(top_recommendations, 1):
            property_name = prop.get('propertyName', 'Unnamed Property')
            type_name = prop.get('typeName', 'Not specified')
            address = prop.get('address', 'Not specified')
            price = prop.get('marketValue', 0)
            property_id = prop.get('id', '')
            property_link = f"{PROPERTY_BASE_URL}{property_id}"

            properties_html += f"""
            <div class="property">
                <h3>{property_name}</h3>
                <div class="property-info">
                    <p><strong>Type:</strong> {type_name}</p>
                    <p><strong>Address:</strong> {address}</p>
                    <p><strong>Price:</strong> ₹{price:,.0f}</p>
                </div>
                <a href="{property_link}" class="btn">View Property</a>
            </div>
            """

        # Create the full email body with embedded CSS
        email_body = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Top Property Recommendations</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 600px;
            margin: 20px auto;
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            padding: 20px;
            background: #fff;
            border-bottom: 1px solid #eee;
        }}
        .logo {{
            max-width: 150px;
            margin-bottom: 10px;
        }}
        .platform-name {{
            font-size: 20px;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 5px;
        }}
        .tagline {{
            font-size: 14px;
            color: #666;
            margin-bottom: 20px;
        }}
        .content {{
            padding: 20px;
            text-align: left;
        }}
        .greeting {{
            font-size: 18px;
            color: #4CAF50;
            margin-bottom: 15px;
        }}
        .intro {{
            margin-bottom: 20px;
            line-height: 1.6;
            color: #555;
        }}
        .property {{
            margin-bottom: 20px;
            padding: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            background: #fff;
        }}
        .property h3 {{
            margin: 0 0 10px 0;
            color: #4CAF50;
            font-size: 16px;
        }}
        .property-info p {{
            margin: 5px 0;
            color: #555;
            font-size: 14px;
        }}
        .btn {{
            display: inline-block;
            padding: 10px 20px;
            background: #4CAF50;
            color: white !important;
            text-decoration: none !important;
            border-radius: 5px;
            font-weight: bold;
            margin-top: 10px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 0 0 8px 8px;
            font-size: 12px;
            color: #666;
            margin-top: 20px;
        }}
        .footer a {{
            color: #4CAF50;
            text-decoration: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <p class="tagline">Where Luxury Meets Lifestyle</p>
        </div>
        <div class="content">
            <p class="greeting">Hi {first_name} {last_name},</p>
            <p class="intro">
                Based on your preferences, we've selected the top properties that match your criteria.
                Explore these exclusive listings and find your dream home!
            </p>
            <h3>🏆 Your Top {len(top_recommendations)} Property Recommendations</h3>
            {properties_html}
        </div>
        <div class="footer">
            <p>Need help? <a href="#">Contact Us</a> | <a href="#">Unsubscribe</a></p>
        </div>
    </div>
</body>
</html>
"""

        params = {
            'toEmail': email,
            'subject': subject,
            'body': email_body
        }

        logger.info(f"Sending email to {email}")
        response = requests.post(EMAIL_API_URL, params=params, timeout=30)

        if response.status_code == 404:
            logger.error(f"Email API endpoint not found (404): {EMAIL_API_URL}")
            return False, f"Email API endpoint not found. Please check the API URL."
        elif response.status_code != 200:
            logger.error(f"Email API returned status {response.status_code}: {response.text}")
            return False, f"Email API error: Status {response.status_code} - {response.text}"

        response.raise_for_status()
        logger.info(f"Email sent successfully to {email}")
        return True, "Email sent successfully"

    except requests.exceptions.Timeout:
        return False, "Email API timeout - service may be slow or unavailable"
    except requests.exceptions.ConnectionError as e:
        return False, "Email API connection failed - service may be down"
    except requests.exceptions.HTTPError as e:
        error_text = e.response.text if e.response else "No response text"
        status_code = e.response.status_code if e.response else "Unknown"
        return False, f"Email API error: Status {status_code} - {error_text}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error sending email: {str(e)}")
        return False, f"Email request failed: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error sending email: {str(e)}")
        return False, f"Failed to send email: {str(e)}"
@app.route('/send_recommendations_email', methods=['POST'])
def send_recommendations_email():

    data = request.get_json()

    first_name = data.get("firstName")
    last_name = data.get("lastName")
    email = data.get("email")
    recommendations = data.get("recommendations", [])

    if not email or not first_name:
        return jsonify({"success": False, "message": "Missing required fields"}), 400

    try:
        # Build recommendation HTML
        properties_html = ""
        for prop in recommendations:
            properties_html += f"""
            <div style="margin-bottom:15px;">
                <h3>{prop.get('propertyName')}</h3>
                <p><strong>Price:</strong> ₹{prop.get('marketValue')}</p>
                <p><strong>Location:</strong> {prop.get('address')}</p>
                <hr>
            </div>
            """

        html_content = f"""
        <html>
        <body style="font-family:Arial;">
            <h2>Hello {first_name},</h2>
            <p>Here are your matched property recommendations:</p>
            {properties_html}
            <br>
            <p>Thank you for using Property AI.</p>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Your Property Recommendations"
        msg["From"] = "your_email@gmail.com"
        msg["To"] = email

        msg.attach(MIMEText(html_content, "html"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login("your_email@gmail.com", "your_app_password")
        server.sendmail(msg["From"], msg["To"], msg.as_string())
        server.quit()

        return jsonify({"success": True, "message": "Email sent successfully!"})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations_route():
    """Get property recommendations with parallel processing optimization and non-blocking property fetch - no timeouts"""
    global active_requests
    
    # Check concurrent request limit
    with request_lock:
        if active_requests >= MAX_CONCURRENT_REQUESTS:
            logger.warning(f"Too many concurrent requests ({active_requests}), rejecting request")
            return jsonify({"error": "Server is busy, please try again later"}), 503
        active_requests += 1
    
    try:
        # Validate request data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        user_input = data.get('description', '')
        price_range = data.get('price_range', None)
        property_type = data.get('property_type', None)
        
        # Handle partial inputs
        if not user_input and not property_type and not price_range:
            return jsonify({"error": "Please provide at least one search criteria"})
        
        # If no description is provided, create one based on available inputs
        if not user_input:
            user_input = f"{property_type if property_type else ''} property"
            if price_range:
                # Handle multiple price ranges
                if isinstance(price_range, list) and len(price_range) > 0:
                    if isinstance(price_range[0], list):
                        # Array of price ranges
                        ranges_text = []
                        for range_item in price_range:
                            ranges_text.append(f"₹{range_item[0]} to ₹{range_item[1]}")
                        user_input += f" within price ranges: {', '.join(ranges_text)}"
                    else:
                        # Single price range
                        user_input += f" within price range ₹{price_range[0]} to ₹{price_range[1]}"
        
        logger.info(f"Search request - Type: {property_type}, Price Range: {price_range}, Description: {user_input}")
        
        # Handle multiple price ranges - convert to min/max arrays
        min_price = None
        max_price = None
        if price_range:
            if isinstance(price_range, list) and len(price_range) > 0:
                if isinstance(price_range[0], list):
                    # Array of price ranges
                    min_price = [range[0] for range in price_range]
                    max_price = [range[1] for range in price_range]
                else:
                    # Single price range
                    min_price = [price_range[0]]
                    max_price = [price_range[1]]
        
        # Check if we have properties in cache first
        try:
            properties = get_cached_properties(
                property_type=property_type,
                min_price=min_price,
                max_price=max_price
            )
        except Exception as e:
            logger.error(f"Error getting cached properties: {str(e)}")
            properties = []
        
        # If no properties found and we're not currently fetching, start background fetch
        if not properties and not is_fetching_properties:
            logger.info("No properties found, initiating background fetch...")
            try:
                threading.Thread(target=lambda: fetch_properties_background_safe(property_type, min_price, max_price), daemon=True).start()
            except Exception as e:
                logger.error(f"Error starting background fetch: {str(e)}")
        
        # Use ThreadPoolExecutor for parallel recommendation processing without timeout
        def process_recommendations():
            try:
                return get_recommendations(user_input, price_range, property_type)
            except Exception as e:
                logger.error(f"Error in process_recommendations: {str(e)}")
                return []
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(process_recommendations)
                recommendations = future.result()  # Removed timeout
            
            # Return empty list if no recommendations found (no fallbacks)
            if not recommendations:
                logger.info("No recommendations found matching criteria, returning empty list")
                recommendations = []
            
            logger.info(f"Returning {len(recommendations)} properties for search using parallel processing")
            return jsonify(recommendations)
        except Exception as e:
            logger.error(f"Error processing recommendations: {str(e)}")
            return jsonify({"error": "An error occurred while processing recommendations"})
    except Exception as e:
        logger.error(f"Unexpected error in get_recommendations_route: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500
    finally:
        # Decrease active request count
        with request_lock:
            active_requests -= 1

@app.route('/refresh_properties', methods=['POST'])
def refresh_properties_route():
    """Manually trigger property refresh (for debugging) - non-blocking"""
    try:
        logger.info("🔄 Manual property refresh triggered")
        # Start background refresh - non-blocking
        threading.Thread(target=refresh_properties_background, daemon=True).start()
        return jsonify({"message": "Property refresh started in background", "status": "success"})
    except Exception as e:
        logger.error(f"Error triggering property refresh: {str(e)}")
        return jsonify({"error": "Failed to trigger property refresh"})

@app.route('/fetch_properties', methods=['POST'])
def fetch_properties_route():
    """Manually fetch and store properties (for debugging) - non-blocking"""
    try:
        logger.info("🔄 Manual property fetch triggered")
        # Start background fetch - non-blocking
        def fetch_and_store():
            try:
                data = fetch_properties_background_safe()
                if data and 'data' in data:
                    new_properties = [p for p in data['data'] if isinstance(p, dict)]
                    if new_properties:
                        # Store properties in ChromaDB
                        update_property_cache(new_properties)
                        update_cache_timestamp()
                        logger.info(f"✅ Manually fetched and stored {len(new_properties)} properties")
                    else:
                        logger.warning("No properties fetched from API")
                else:
                    logger.warning("Failed to fetch properties from API")
            except Exception as e:
                logger.error(f"Error in manual property fetch: {str(e)}")
        
        threading.Thread(target=fetch_and_store, daemon=True).start()
        return jsonify({
            "message": "Property fetch started in background",
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error in manual property fetch: {str(e)}")
        return jsonify({"error": f"Failed to fetch properties: {str(e)}"})

@app.route('/cache_status', methods=['GET'])
def cache_status_route():
    """Get cache status information"""
    try:
        property_count = check_property_cache_status()
        cache_stale = is_cache_stale()
        
        status_info = {
            "property_count": property_count,
            "cache_stale": cache_stale,
            "cache_expiry_hours": CACHE_EXPIRY_SECONDS / 3600
        }
        
        if os.path.exists(CACHE_TIMESTAMP_PATH):
            with open(CACHE_TIMESTAMP_PATH, 'r') as f:
                last_time = float(f.read().strip())
            status_info["last_update"] = datetime.fromtimestamp(last_time).isoformat()
        else:
            status_info["last_update"] = None
            
        return jsonify(status_info)
    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        return jsonify({"error": "Failed to get cache status"})
@app.route('/all_properties', methods=['GET'])
def all_properties_route():
    """Return all properties currently cached in ChromaDB (no filters)."""
    try:
        # get_cached_properties() already pulls from ChromaDB and converts metadata
        properties = get_cached_properties()
        return jsonify(properties)
    except Exception as e:
        logger.error(f"Error fetching all properties: {str(e)}")
        return jsonify({"error": "Failed to load properties"}), 500
@app.route("/search_suggestions")
def search_suggestions():
    query = request.args.get("q", "").lower()

    properties = get_cached_properties()

    matches = []

    for p in properties:
        text_blob = " ".join([
            str(p.get("propertyName", "")),
            str(p.get("address", "")),
            str(p.get("typeName", "")),
            str(p.get("description", "")),
            " ".join(p.get("features", []))
        ]).lower()

        if query in text_blob:
            matches.append({
                "propertyName": p.get("propertyName"),
                "address": p.get("address")
            })

        if len(matches) >= 5:
            break

    return jsonify(matches)
@app.route("/search")
def search():
    query = request.args.get("q", "").lower()
    properties = get_cached_properties()

    matched = []

    for p in properties:
        text_blob = " ".join([
            str(p.get("propertyName", "")),
            str(p.get("address", "")),
            str(p.get("typeName", "")),
            str(p.get("description", "")),
            " ".join(p.get("features", []))
        ]).lower()

        if query in text_blob:
            matched.append(p)

    return jsonify(matched)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring multi-user performance"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        health_info = {
            "status": "healthy",
            "active_requests": active_requests,
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "timestamp": datetime.now().isoformat()
        }
        
        # Check if system is overloaded
        if active_requests >= MAX_CONCURRENT_REQUESTS * 0.8:
            health_info["status"] = "warning"
        if cpu_percent > 80 or memory.percent > 80:
            health_info["status"] = "warning"
            
        return jsonify(health_info)
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({"status": "error", "error": str(e)})

def display_countdown():
    """Display countdown in terminal and manage scheduled property refresh"""
    global stop_countdown, countdown_initialized
    
    # Initialize countdown properly
    if not countdown_initialized:
        logger.info("🔄 Initializing countdown system...")
        # Force initial property fetch if no cache exists - non-blocking
        if not os.path.exists(CACHE_TIMESTAMP_PATH):
            logger.info("🔄 No cache timestamp found, performing initial property fetch...")
            # Start initial fetch in background thread
            threading.Thread(target=force_initial_property_fetch, daemon=True).start()
        countdown_initialized = True
    
    last_display_time = 0
    display_interval = 60  # Update display every 60 seconds
    
    while not stop_countdown:
        try:
            current_time = time.time()
            
            # Only update display every minute to reduce spam
            if current_time - last_display_time >= display_interval:
                if os.path.exists(CACHE_TIMESTAMP_PATH):
                    with open(CACHE_TIMESTAMP_PATH, 'r') as f:
                        last_time = float(f.read().strip())
                    
                    time_diff = current_time - last_time
                    time_remaining = CACHE_EXPIRY_SECONDS - time_diff
                    
                    if time_remaining > 0:
                        # Clear the current line and print countdown
                        sys.stdout.write('\r' + ' ' * 100)
                        sys.stdout.write('\r')
                        next_update = datetime.fromtimestamp(current_time + time_remaining)
                        sys.stdout.write(f"⏰ Next data fetch in: {format_time_remaining(time_remaining)} (at {next_update.strftime('%H:%M:%S')})")
                        sys.stdout.flush()
                        last_display_time = current_time
                    else:
                        # Time to refresh properties - non-blocking
                        logger.info("🔄 24 hours have passed, starting scheduled property refresh...")
                        threading.Thread(target=refresh_properties_background, daemon=True).start()
                        sys.stdout.write('\r' + ' ' * 100)
                        sys.stdout.write('\r')
                        sys.stdout.write("✅ Scheduled property refresh initiated")
                        sys.stdout.flush()
                        last_display_time = current_time
                        # Wait 60 seconds after initiating refresh before continuing
                        time.sleep(60)
                else:
                    # No cache timestamp - try to create one - non-blocking
                    sys.stdout.write('\r' + ' ' * 100)
                    sys.stdout.write('\r')
                    sys.stdout.write("🔄 No cache timestamp found. Attempting initial fetch...")
                    sys.stdout.flush()
                    last_display_time = current_time
                    
                    # Try to fetch properties and create timestamp - non-blocking
                    threading.Thread(target=force_initial_property_fetch, daemon=True).start()
                    sys.stdout.write('\r' + ' ' * 100)
                    sys.stdout.write('\r')
                    sys.stdout.write("✅ Initial property fetch initiated")
                    sys.stdout.flush()
                    time.sleep(300)  # Wait 5 minutes before checking again
                    continue
            
            # Sleep for a shorter interval
            time.sleep(10)  # Check every 10 seconds instead of 1 second
            
        except Exception as e:
            logger.error(f"Error in countdown display: {e}")
            time.sleep(30)  # Wait 30 seconds on error before retrying

def start_countdown():
    """Start the countdown display thread"""
    global countdown_thread, stop_countdown, countdown_initialized
    stop_countdown = False
    countdown_initialized = False
    countdown_thread = threading.Thread(target=display_countdown, daemon=True)
    countdown_thread.start()
    logger.info("✅ Countdown thread started")

def stop_countdown_display():
    """Stop the countdown display thread"""
    global stop_countdown
    stop_countdown = True
    if countdown_thread:
        countdown_thread.join(timeout=1)
    logger.info("✅ Countdown thread stopped")

def get_system_info():
    """Get system information for performance optimization"""
    import psutil
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    logger.info(f"🚀 System Info - CPU Cores: {cpu_count}, Memory: {memory_gb:.1f}GB")
    return cpu_count, memory_gb

def check_property_cache_status():
    """Check the status of property cache and log information"""
    global property_collection
    try:
        if property_collection:
            results = property_collection.get()
            if results and results['ids']:
                logger.info(f"📊 Property Cache Status: {len(results['ids'])} properties available")
                return len(results['ids'])
            else:
                logger.warning("📊 Property Cache Status: No properties in cache")
                return 0
        else:
            logger.warning("📊 Property Cache Status: Collection not initialized")
            return 0
    except Exception as e:
        logger.error(f"Error checking property cache status: {e}")
        return 0

def ensure_properties_available():
    """Ensure properties are available in cache, fetch if needed - non-blocking"""
    global property_collection
    try:
        if not property_collection:
            logger.info("Collection not initialized, initializing...")
            initialize_collection()
            return
        
        results = property_collection.get()
        if not results or not results['ids']:
            logger.info("🔄 No properties in cache, fetching from API...")
            # Force fetch properties - non-blocking
            threading.Thread(target=force_initial_property_fetch, daemon=True).start()
            logger.info("✅ Property fetch initiated in background")
        else:
            logger.info(f"✅ {len(results['ids'])} properties already available in cache")
    except Exception as e:
        logger.error(f"Error ensuring properties are available: {e}")

if __name__ == '__main__':
    try:
        # Get system information
        cpu_count, memory_gb = get_system_info()
        
        logger.info(f"🚀 Starting Property Recommendation System with parallel processing optimization")
        logger.info(f"🔧 Using {min(cpu_count, 8)} workers for parallel processing")
        
        # Ensure properties are available in cache - non-blocking
        ensure_properties_available()
        
        # Check property cache status
        property_count = check_property_cache_status()
        if property_count == 0:
            logger.warning("⚠️ Still no properties in cache after initialization")
            # Try one more time to fetch properties - non-blocking
            threading.Thread(target=force_initial_property_fetch, daemon=True).start()
            logger.info("✅ Property fetch retry initiated in background")
        else:
            logger.info(f"✅ Cache has {property_count} properties ready for recommendations")
        
        # Start the countdown display
        start_countdown()
        
        # Change port to 7860 for Hugging Face Spaces with multi-user support
        app.run(debug=False, host='0.0.0.0', port=7860, threaded=True)
    except KeyboardInterrupt:
        logger.info("🛑 Application interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error starting application: {str(e)}")
    finally:
        # Stop the countdown display when the app stops
        stop_countdown_display()
        logger.info("🛑 Application stopped")