# PageIndex - Architecture & Multi-User Scaling Guide

## Overview

**PageIndex** is a **vectorless, reasoning-based RAG (Retrieval-Augmented Generation)** system that transforms PDF/Markdown documents into hierarchical tree structures optimized for LLM reasoning-based retrieval.

### Core Philosophy

Traditional RAG systems use vector embeddings + similarity search. The problem: **similarity != relevance**.

PageIndex takes a different approach:
1. Build a "Table of Contents" tree structure from documents
2. Use LLM reasoning to navigate the tree (like a human expert would)
3. Retrieve relevant sections through tree search, not vector similarity

**Result:** 98.7% accuracy on FinanceBench vs traditional RAG systems.

---

## Architecture

```
                    +------------------+
                    |   PDF/Markdown   |
                    +--------+---------+
                             |
                             v
              +------------------------------+
              |       Document Parser        |
              |  (PyPDF2/PyMuPDF/Regex MD)   |
              +-------------+----------------+
                            |
                            v
              +------------------------------+
              |      TOC Detection/Gen       |
              |  (LLM-based, 20 pages scan)  |
              +-------------+----------------+
                            |
                            v
              +------------------------------+
              |    Tree Structure Builder    |
              | (Hierarchical JSON with IDs) |
              +-------------+----------------+
                            |
                            v
              +------------------------------+
              |      Summary Generator       |
              |  (Async parallel per node)   |
              +-------------+----------------+
                            |
                            v
              +------------------------------+
              |    Tree Structure (JSON)     |
              |   {title, node_id, summary,  |
              |    start_index, end_index,   |
              |    nodes[]}                  |
              +------------------------------+
```

---

## Core Components

### 1. Document Parser (`utils.py`)

```python
# PDF Parsing
def get_page_tokens(pdf_path, model):
    """Extract text and token count from each page"""
    # Uses PyPDF2 or PyMuPDF
    # Returns: [(page_text, token_count), ...]
```

**Key Functions:**
- `count_tokens(text, model)` - Token counting via tiktoken
- `ChatGPT_API(model, prompt)` - Sync API calls (10 retry)
- `ChatGPT_API_async(model, prompt)` - Async API calls
- `extract_json(content)` - Parse JSON from LLM responses

### 2. TOC Detection (`page_index.py`)

The system scans the first 20 pages (configurable) to detect existing Table of Contents:

```python
def check_toc(pdf_path, opt):
    """
    Detect TOC in document
    Returns one of three processing modes:
    - process_toc_with_page_numbers
    - process_toc_no_page_numbers  
    - process_no_toc
    """
```

**Detection Flow:**
1. `toc_detector_single_page()` - LLM checks each page for TOC presence
2. `find_toc_pages()` - Scans consecutive TOC pages
3. `detect_page_index()` - Determines if TOC has page numbers
4. `toc_transformer()` - Converts raw TOC to structured JSON

### 3. Tree Structure Builder

When no TOC exists, the system generates one using LLM:

```python
def tree_parser(page_text, model):
    """Generate hierarchical structure from content"""
    # Uses LLM to identify sections/subsections
    # Returns JSON tree structure
```

**Node Structure:**
```json
{
  "title": "Section Name",
  "node_id": "0001",
  "start_index": 1,
  "end_index": 5,
  "summary": "LLM-generated summary",
  "nodes": [/* child nodes */]
}
```

### 4. Large Node Processing

Nodes exceeding limits are recursively split:

```python
def process_large_node_recursively(node, page_list, opt):
    """
    Split large nodes based on:
    - max_page_num_each_node: 10 (default)
    - max_token_num_each_node: 20000 (default)
    """
```

### 5. Verification & Correction

```python
async def verify_toc(structure, page_list, model):
    """Verify titles appear on claimed pages"""
    # Concurrent async verification
    
def fix_incorrect_toc(structure, page_list, model):
    """Auto-correct wrong page references"""
    # Up to 3 retry attempts
```

### 6. Markdown Processing (`page_index_md.py`)

```python
async def md_to_tree(md_path, **options):
    """
    Process markdown to tree structure
    1. extract_nodes_from_markdown() - Parse headers
    2. build_tree_from_nodes() - Build hierarchy
    3. tree_thinning_for_index() - Merge small nodes (optional)
    4. generate_summaries_for_structure_md() - Add summaries
    """
```

---

## Configuration

### Default Config (`pageindex/config.yaml`)

```yaml
model: "gpt-4o-2024-11-20"
toc_check_page_num: 20        # Pages to scan for TOC
max_page_num_each_node: 10    # Max pages per node
max_token_num_each_node: 20000 # Max tokens per node
if_add_node_id: "yes"         # Add unique 4-digit IDs
if_add_node_summary: "yes"    # Generate summaries
if_add_doc_description: "no"  # Document-level description
if_add_node_text: "no"        # Include raw text in output
```

### CLI Usage

```bash
# Process PDF
python run_pageindex.py \
    --pdf_path ./docs/report.pdf \
    --model gpt-4o-2024-11-20 \
    --max-pages-per-node 10 \
    --max-tokens-per-node 20000 \
    --if-add-node-summary yes

# Process Markdown
python run_pageindex.py \
    --md_path ./docs/guide.md \
    --if-thinning yes \
    --thinning-threshold 5000
```

---

## Current Limitations (Single-User)

The current implementation is designed for **local, single-user processing**:

| Aspect | Current State |
|--------|---------------|
| Storage | Local JSON files (`./results/`) |
| User Isolation | None |
| API Keys | Single key from `.env` |
| Queue System | None (synchronous CLI) |
| Caching | None |
| Rate Limiting | Basic retry (10x with 1s delay) |

---

## Multi-User Scaling Strategy

### Architecture for Multi-Tenant System

```
                                    +------------------+
                                    |   API Gateway    |
                                    |   (FastAPI)      |
                                    +--------+---------+
                                             |
                    +------------------------+------------------------+
                    |                        |                        |
           +--------v--------+      +--------v--------+      +--------v--------+
           |  Auth Service   |      |  Job Queue      |      |  Rate Limiter   |
           |  (JWT/OAuth)    |      |  (Celery/Redis) |      |  (per user)     |
           +-----------------+      +--------+--------+      +-----------------+
                                             |
                    +------------------------+------------------------+
                    |                        |                        |
           +--------v--------+      +--------v--------+      +--------v--------+
           |  Document       |      |  PageIndex      |      |  Result Store   |
           |  Storage (S3)   |      |  Workers        |      |  (MongoDB)      |
           +-----------------+      +-----------------+      +-----------------+
```

### 1. API Layer (FastAPI)

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()

@app.post("/documents/process")
async def process_document(
    file: UploadFile,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    1. Save document to S3 with user prefix
    2. Create job in queue
    3. Return job_id for polling
    """
    doc_path = await save_to_s3(file, user.id)
    job = await create_processing_job(doc_path, user.id)
    return {"job_id": job.id, "status": "queued"}

@app.get("/documents/{doc_id}/tree")
async def get_document_tree(
    doc_id: str,
    user: User = Depends(get_current_user)
):
    """Retrieve processed tree structure"""
    tree = await get_tree_from_db(doc_id, user.id)
    if not tree:
        raise HTTPException(404, "Document not found")
    return tree
```

### 2. Job Queue (Celery + Redis)

```python
# workers/tasks.py
from celery import Celery
from pageindex import page_index_main

celery = Celery('pageindex', broker='redis://localhost:6379')

@celery.task(bind=True, max_retries=3)
def process_document_task(self, doc_path: str, user_id: str, options: dict):
    """
    Background worker for document processing
    """
    try:
        # Load document from S3
        local_path = download_from_s3(doc_path)
        
        # Process with PageIndex
        opt = create_options(options)
        tree = page_index_main(local_path, opt)
        
        # Store result in MongoDB
        save_tree_to_db(tree, user_id, doc_path)
        
        # Cleanup
        os.remove(local_path)
        
        return {"status": "completed", "doc_id": doc_path}
    except Exception as e:
        self.retry(exc=e, countdown=60)
```

### 3. Database Schema (MongoDB)

```python
# models/document.py
from pydantic import BaseModel
from datetime import datetime

class DocumentTree(BaseModel):
    """MongoDB document schema"""
    _id: str                      # doc_id
    user_id: str                  # tenant isolation
    original_filename: str
    s3_path: str
    tree_structure: dict          # The PageIndex output
    metadata: dict = {
        "model_used": str,
        "processing_time_seconds": float,
        "total_pages": int,
        "total_nodes": int
    }
    created_at: datetime
    updated_at: datetime
    status: str                   # queued, processing, completed, failed

# Indexes
# db.documents.createIndex({"user_id": 1, "created_at": -1})
# db.documents.createIndex({"user_id": 1, "status": 1})
```

### 4. Document Storage (S3)

```python
# storage/s3.py
import boto3
from botocore.config import Config

class DocumentStorage:
    def __init__(self):
        self.s3 = boto3.client('s3', config=Config(
            retries={'max_attempts': 3}
        ))
        self.bucket = os.getenv('S3_BUCKET')
    
    async def upload(self, file: UploadFile, user_id: str) -> str:
        """Upload with user prefix for isolation"""
        key = f"documents/{user_id}/{uuid4()}/{file.filename}"
        await self.s3.upload_fileobj(file.file, self.bucket, key)
        return key
    
    async def download(self, key: str) -> str:
        """Download to temp file for processing"""
        local_path = f"/tmp/{uuid4()}"
        await self.s3.download_file(self.bucket, key, local_path)
        return local_path
```

### 5. Rate Limiting (Per User)

```python
# middleware/rate_limit.py
from fastapi import Request
import redis.asyncio as redis

class RateLimiter:
    def __init__(self):
        self.redis = redis.from_url("redis://localhost:6379")
    
    async def check_limit(self, user_id: str, limit: int = 10, window: int = 3600):
        """
        Rate limit per user per hour
        - Free tier: 10 docs/hour
        - Pro tier: 100 docs/hour
        - Enterprise: unlimited
        """
        key = f"rate:{user_id}:{int(time.time() // window)}"
        current = await self.redis.incr(key)
        
        if current == 1:
            await self.redis.expire(key, window)
        
        if current > limit:
            raise HTTPException(429, "Rate limit exceeded")
```

### 6. OpenAI API Key Management

```python
# config/api_keys.py
class APIKeyManager:
    """
    Multi-user API key strategies:
    1. Shared pool (default) - Platform pays, rate limit users
    2. BYOK (Bring Your Own Key) - User provides their key
    3. Enterprise - Dedicated key per organization
    """
    
    def get_api_key(self, user: User) -> str:
        if user.subscription == "enterprise":
            return user.organization.openai_key
        elif user.has_own_key:
            return decrypt(user.openai_key)
        else:
            return self.get_pool_key()
    
    def get_pool_key(self) -> str:
        """Round-robin through pool of API keys"""
        keys = os.getenv('OPENAI_API_KEYS').split(',')
        return keys[self.counter.increment() % len(keys)]
```

---

## Scaling Considerations

### Cost Management

| Document Size | Est. Tokens | Est. Cost (GPT-4o) |
|--------------|-------------|-------------------|
| 10 pages | ~15K tokens | ~$0.15 |
| 50 pages | ~75K tokens | ~$0.75 |
| 200 pages | ~300K tokens | ~$3.00 |

**Cost Optimization Strategies:**
1. Cache processed documents
2. Use GPT-4o-mini for initial TOC detection
3. Batch similar operations
4. Implement document deduplication

### Performance Optimization

```python
# Existing async patterns in PageIndex
async def verify_toc_concurrent(structure, page_list, model):
    """Already uses asyncio.gather() for parallel verification"""
    tasks = [check_title_appearance(item, page_list, model) for item in structure]
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

**Additional Optimizations:**
1. **Document Caching:** Store processed trees, invalidate on document update
2. **Result Pagination:** For large trees, paginate node responses
3. **Streaming:** Stream partial results during long processing
4. **Worker Autoscaling:** Scale Celery workers based on queue depth

### High Availability

```yaml
# docker-compose.yml example
services:
  api:
    image: pageindex-api
    replicas: 3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
  
  worker:
    image: pageindex-worker
    replicas: 5
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
  
  mongodb:
    image: mongo:7
    volumes:
      - mongo-data:/data/db
```

---

## Migration Path

### Phase 1: Add Database Layer
- [ ] Add MongoDB for storing processed trees
- [ ] Add user_id field to all operations
- [ ] Keep CLI working for backward compatibility

### Phase 2: Add API Server
- [ ] FastAPI server with authentication
- [ ] Basic job queue with Celery
- [ ] S3 storage for documents

### Phase 3: Production Hardening
- [ ] Rate limiting per user
- [ ] Monitoring and alerting
- [ ] Auto-scaling workers
- [ ] Multi-region deployment

### Phase 4: Enterprise Features
- [ ] BYOK support
- [ ] SSO integration
- [ ] Audit logging
- [ ] Data retention policies

---

## Example: Minimal Multi-User Setup

```python
# Minimal FastAPI + MongoDB setup
from fastapi import FastAPI, UploadFile, BackgroundTasks
from motor.motor_asyncio import AsyncIOMotorClient
from pageindex import page_index_main
import tempfile

app = FastAPI()
db = AsyncIOMotorClient("mongodb://localhost:27017").pageindex

@app.post("/process")
async def process(
    file: UploadFile,
    user_id: str,
    background_tasks: BackgroundTasks
):
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    # Create job record
    job_id = str(uuid4())
    await db.jobs.insert_one({
        "_id": job_id,
        "user_id": user_id,
        "filename": file.filename,
        "status": "processing"
    })
    
    # Process in background
    background_tasks.add_task(
        process_and_store, 
        tmp_path, 
        job_id, 
        user_id
    )
    
    return {"job_id": job_id}

async def process_and_store(path: str, job_id: str, user_id: str):
    try:
        opt = config(model="gpt-4o-2024-11-20", ...)
        tree = page_index_main(path, opt)
        
        await db.trees.insert_one({
            "_id": job_id,
            "user_id": user_id,
            "tree": tree
        })
        await db.jobs.update_one(
            {"_id": job_id},
            {"$set": {"status": "completed"}}
        )
    finally:
        os.unlink(path)
```

---

## Summary

PageIndex is well-architected for single-user local processing with good async patterns already in place. Scaling to multi-user requires:

1. **Database Layer** - MongoDB for trees + user isolation
2. **Job Queue** - Celery/Redis for async processing
3. **Storage** - S3 for documents with user prefixes
4. **API Server** - FastAPI with auth/rate limiting
5. **Cost Management** - Per-user limits, caching, key pooling

The existing async/await patterns and modular code make this migration relatively straightforward.
