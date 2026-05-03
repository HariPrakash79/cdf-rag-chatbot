# Retrieval Tuning Results

| Config | Accuracy | Grounding | Citation | Latency |
|--------|----------|-----------|----------|---------|
| chunk=1500_k=4 | 81.1% | 90.2% | 52.8% | 1.63s |
| chunk=1500_k=12 | 89.2% | 96.2% | 52.8% | 1.53s |

**Best config:** `chunk=1500_k=12`

**Recommended changes:**
- `build_index.py`: `chunk_size=1500`
- `app.py` retriever: `k=12`, `fetch_k=40`