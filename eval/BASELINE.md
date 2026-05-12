# Retrieval Tuning Results

| Config | Accuracy | Grounding | Citation | Latency |
|--------|----------|-----------|----------|---------|
| chunk=500_k=8 | 86.0% | 92.7% | 69.5% | 1.53s |
| chunk=1000_k=8 (baseline) | 86.6% | 92.8% | 69.5% | 1.46s |
| chunk=1500_k=8 | 82.9% | 90.2% | 69.5% | 1.46s |

**Best config:** `chunk=1000_k=8 (baseline)`

**Recommended changes:**
- `build_index.py`: `chunk_size=1000`
- `app.py` retriever: `k=8`, `fetch_k=30`