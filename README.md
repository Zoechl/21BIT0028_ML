# 1. Backend with Flask:

from flask import Flask, request, jsonify
import time
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from threading import Thread
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Rate limiter: max 5 requests per user
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["5 per minute"]
)

# Initialize Pinecone and OpenAI Embeddings
pinecone.init(api_key="your_pinecone_api_key", environment="gcp-starter")
index_name = "tk-policy"
if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)
pinecone.create_index(name=index_name, dimension=1536)
index = pinecone.Index(index_name)

embed = OpenAIEmbeddings(openai_api_key="your_openai_api_key")
vectorstore = Pinecone(index, embed.embed_query, text_field="text")

# Background process to scrape news articles
def background_scraper():
    while True:
        # Dummy scraper
        logging.info("Scraping news articles...")
        time.sleep(3600)  # Scrape every hour

scraping_thread = Thread(target=background_scraper)
scraping_thread.start()

# Health endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "API is active!"}), 200

# Search endpoint with caching and rate limiting
@app.route('/search', methods=['POST'])
@limiter.limit("5/minute", override_defaults=False)
def search():
    start_time = time.time()
    data = request.json
    user_id = data.get("user_id")
    text = data.get("text", "")
    top_k = data.get("top_k", 5)
    threshold = data.get("threshold", 0.5)

    # Tokenize the input text, embed it, and query Pinecone
    query_embedding = embed.embed_query(text)
    search_results = index.query(queries=[query_embedding], top_k=top_k, filter={'score': {'$gte': threshold}})
    
    inference_time = time.time() - start_time
    logging.info(f"Request handled in {inference_time} seconds")

    return jsonify({"results": search_results, "inference_time": inference_time}), 200

# Dockerfile - Include this in your Docker setup
# Dockerize the app with Flask and Pinecone
"""
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
# 2. Dockerization (Dockerfile):
# Dockerfile to build the Flask app
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port 5000 for Flask
EXPOSE 5000

CMD ["python", "app.py"]
# 3. Caching with Redis (Optional):

from redis import Redis
import pickle

cache = Redis(host='localhost', port=6379, db=0)

@app.route('/search', methods=['POST'])
@limiter.limit("5/minute", override_defaults=False)
def search():
    user_id = request.json.get('user_id')
    cache_key = f"{user_id}_{request.json.get('text')}"
    
    # Check if cached response exists
    cached_result = cache.get(cache_key)
    if cached_result:
        return pickle.loads(cached_result), 200
    
      # Regular processing...
    query_embedding = embed.embed_query(text)
    search_results = index.query(queries=[query_embedding], top_k=top_k, filter={'score': {'$gte': threshold}})
    
    # Cache the result
    cache.setex(cache_key, 3600, pickle.dumps(search_results))  # Cache for 1 hour
    
    return jsonify(search_results), 200
# 4. Rate Limiting:

@app.route('/search', methods=['POST'])
@limiter.limit("5/minute", override_defaults=False)
def search():
    user_id = request.json.get('user_id')
    # Increment user API call frequency in DB
    user_record = db.get_user(user_id)
    if user_record and user_record.api_call_count >= 5:
        return jsonify({"error": "Too many requests"}), 429
    # Normal processing
# 5. Logging:
import logging
logging.basicConfig(level=logging.INFO)

@app.route('/search', methods=['POST'])
def search():
    logging.info(f"Received search request from user: {request.json.get('user_id')}")
    # Process request and return response
logging.basicConfig(level=logging.INFO)

@app.route('/search', methods=['POST'])
def search():
    logging.info(f"Received search request from user: {request.json.get('user_id')}")
    # Process request and return response
