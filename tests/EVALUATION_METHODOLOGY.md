# Edalo Evaluation Methodology

## How the Comparison Works

The evaluation compares LLM performance **without Edalo processing** (raw data) vs **with Edalo processing** (optimized data) using the same query and the same LLM model.

## Test Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    TEST DATASET                              │
│         328 reviews across 16 files (4 languages)            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │     PROCESS THROUGH EDALO PIPELINE    │
        │  - Extract text                        │
        │  - Clean & deduplicate                 │
        │  - Chunk text                          │
        │  - Generate embeddings                 │
        │  - Store in vector database            │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │         EVALUATION PHASE               │
        └───────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────────────┐          ┌──────────────────────┐
│   RAW DATA TEST      │          │  OPTIMIZED DATA TEST │
│  (Without Edalo)     │          │   (With Edalo)       │
└──────────────────────┘          └──────────────────────┘
```

## 1. Raw Data Test (Without Edalo Processing)

### Process
1. **Data Preparation**:
   - Takes all raw text from all processed documents
   - Combines all text into a single large context string
   - No processing, cleaning, or optimization

2. **LLM Query**:
   - Creates prompt: `"{query}\n\nContext:\n{all_raw_text}"`
   - Sends **entire dataset** to LLM in one request
   - Example: 22,581 input tokens (all 328 reviews)

3. **Metrics Collected**:
   - **Input Tokens**: All raw text sent to LLM
   - **Output Tokens**: LLM response tokens
   - **Total Tokens**: Input + Output
   - **Latency**: Time from query start to response completion
   - **Response**: Full LLM response text

### Code Implementation
```python
def evaluate_raw_data(documents: List[ProcessedDocument], query: str):
    # Combine ALL raw text from all documents
    combined_text = ""
    for doc in documents:
        combined_text += doc.raw_text + "\n\n"
    
    # Send everything to LLM
    prompt = f"{query}\n\nContext:\n{combined_text}"
    result = query_ollama(prompt)
    
    # Count tokens and measure latency
    token_count = count_tokens_approx(prompt)
    result["input_tokens"] = token_count
    result["total_tokens"] = token_count + result.get("tokens", 0)
    return result
```

### Characteristics
- **Input Size**: Very large (all documents)
- **Context Quality**: Unprocessed, may contain duplicates and noise
- **Relevance**: No filtering - all data sent regardless of relevance
- **Token Usage**: High (22,581 input tokens in test)
- **Latency**: Slow (209.53 seconds in test)

## 2. Optimized Data Test (With Edalo Processing)

### Process
1. **Data Preparation** (Done by Edalo Pipeline):
   - Text extracted and cleaned
   - Duplicates removed
   - Text chunked into manageable pieces
   - Embeddings generated for each chunk
   - Stored in vector database (ChromaDB)

2. **RAG (Retrieval-Augmented Generation)**:
   - Query is converted to embedding using the same model
   - Vector database is queried for **top 5 most relevant chunks**
   - Only relevant chunks are retrieved based on semantic similarity
   - Example: 629 input tokens (only 5 relevant chunks)

3. **LLM Query**:
   - Creates prompt: `"{query}\n\nContext:\n{relevant_chunks_only}"`
   - Sends **only relevant data** to LLM
   - Much smaller context, but highly relevant

4. **Metrics Collected**:
   - **Input Tokens**: Only relevant chunks sent to LLM
   - **Output Tokens**: LLM response tokens
   - **Total Tokens**: Input + Output
   - **Latency**: Time from query start to response completion
   - **Response**: Full LLM response text
   - **Context Chunks**: Number of chunks retrieved (typically 5)

### Code Implementation
```python
def evaluate_optimized_data(documents: List[ProcessedDocument], query: str):
    # Get vector database and embedding model
    _, collection = get_vector_db()
    model = load_embedding_model()
    
    # Convert query to embedding
    query_embedding = model.encode([query])[0].tolist()
    
    # Retrieve only top 5 relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,  # Only top 5 most relevant
    )
    
    # Build context from relevant chunks only
    context_texts = results["documents"][0]
    context = "\n\n".join(context_texts)
    
    # Send only relevant data to LLM
    prompt = f"{query}\n\nContext:\n{context}"
    result = query_ollama(prompt)
    
    # Count tokens and measure latency
    token_count = count_tokens_approx(prompt)
    result["input_tokens"] = token_count
    result["total_tokens"] = token_count + result.get("tokens", 0)
    return result
```

### Characteristics
- **Input Size**: Small (only relevant chunks)
- **Context Quality**: Cleaned, deduplicated, structured
- **Relevance**: Highly filtered - only semantically relevant data
- **Token Usage**: Low (629 input tokens in test)
- **Latency**: Fast (35.93 seconds in test)

## 3. Comparison Metrics

### Token Reduction
```python
token_reduction = (
    (raw_results["total_tokens"] - optimized_results["total_tokens"])
    / raw_results["total_tokens"]
    * 100
)
```
- **Raw**: 27,146 total tokens
- **Optimized**: 1,676 total tokens
- **Reduction**: 93.83%

### Latency Reduction
```python
latency_reduction = (
    (raw_results["latency"] - optimized_results["latency"])
    / raw_results["latency"]
    * 100
)
```
- **Raw**: 209.53 seconds
- **Optimized**: 35.93 seconds
- **Reduction**: 82.85%

### Relevance Score
```python
# Compare response to ground truth using cosine similarity
raw_relevance = compute_relevance_score(raw_results["response"], ground_truth)
optimized_relevance = compute_relevance_score(optimized_results["response"], ground_truth)
relevance_increase = optimized_relevance - raw_relevance
```
- **Method**: Cosine similarity between response and ground truth embeddings
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Raw Relevance**: 0.227 (low - generic response)
- **Optimized Relevance**: 0.639 (high - relevant response)
- **Increase**: +0.412 (+41.15% improvement)

## 4. Ground Truth

The evaluation uses a **ground truth** answer to measure response quality:

```python
ground_truth = "Students express concerns about too much theoretical content and insufficient practical exercises. They want more hands-on activities and fewer lectures."
```

This represents the **expected answer** to the query:
```
Query: "What are the main issues students mention in their feedback?"
```

The relevance score measures how well each response (raw vs optimized) matches this ground truth.

## 5. Results Presentation

### Metrics JSON
All results are stored in `output/metrics.json`:
```json
{
  "raw": {
    "response": "...",
    "latency": 209.53,
    "total_tokens": 27146,
    "relevance": 0.227
  },
  "optimized": {
    "response": "...",
    "latency": 35.93,
    "total_tokens": 1676,
    "relevance": 0.639,
    "context_chunks": 5
  },
  "improvement": {
    "token_reduction_percent": 93.83,
    "latency_reduction_percent": 82.85,
    "relevance_increase": 0.412
  }
}
```

### Visualization
A bar chart is generated showing:
- Token Reduction (%)
- Latency Reduction (%)
- Relevance Increase (%)

### Console Output
Test results are printed to console:
```
Evaluation Metrics:
  Raw data - Tokens: 27146, Latency: 209.53s
  Optimized data - Tokens: 1676, Latency: 35.93s

Improvements:
  Token reduction: 93.83%
  Latency reduction: 82.85%
  Relevance increase: 0.4115
```

## Key Differences Summary

| Aspect | Raw Data (Without Edalo) | Optimized Data (With Edalo) |
|--------|-------------------------|----------------------------|
| **Data Preparation** | None - raw text only | Cleaned, deduplicated, chunked |
| **Context Size** | All documents (22,581 tokens) | Top 5 relevant chunks (629 tokens) |
| **Retrieval Method** | None - everything sent | Semantic search (RAG) |
| **Token Usage** | High (27,146 tokens) | Low (1,676 tokens) |
| **Latency** | Slow (209.53s) | Fast (35.93s) |
| **Relevance** | Low (0.227) - generic response | High (0.639) - relevant response |
| **Response Quality** | Generic ML advice | Specific student feedback insights |

## Why This Comparison is Valid

1. **Same Dataset**: Both tests use the same 328 reviews
2. **Same Query**: Both tests use the same question
3. **Same LLM**: Both tests use the same model (deepseek-coder:6.7b)
4. **Same Metrics**: Both tests measure the same things (tokens, latency, relevance)
5. **Fair Comparison**: Only difference is data preparation and retrieval method

## Test Results Interpretation

### Token Reduction (93.83%)
- **Meaning**: Edalo reduces token usage by 93.83%
- **Impact**: Lower costs, faster processing, more efficient
- **Cause**: RAG retrieves only relevant chunks instead of all data

### Latency Reduction (82.85%)
- **Meaning**: Edalo reduces response time by 82.85%
- **Impact**: Near-real-time responses, better user experience
- **Cause**: Smaller context = faster LLM processing

### Relevance Increase (+41.15%)
- **Meaning**: Edalo improves response relevance by 41.15%
- **Impact**: More accurate, useful responses
- **Cause**: RAG retrieves semantically relevant chunks, not random data

## Limitations

1. **Ground Truth**: Requires manual creation of expected answer
2. **Single Query**: Tests with one query type (could test multiple)
3. **Cosine Similarity**: Relevance measured by semantic similarity, not human evaluation
4. **Token Counting**: Uses approximation (1 token ≈ 4 characters)
5. **LLM Variability**: Results may vary with different LLM models

## Future Improvements

1. **Multiple Queries**: Test with different query types
2. **Human Evaluation**: Add human evaluation alongside automated metrics
3. **More Metrics**: Add precision, recall, F1 score
4. **A/B Testing**: Test with different chunk sizes, retrieval methods
5. **Cost Analysis**: Calculate actual cost savings in dollars

