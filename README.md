# 🤝 CDF RAG Chatbot

An internal AI assistant for Community Dreams Foundation volunteers and staff. Answers organization-wide questions — onboarding, policies, payments, and workflows — grounded strictly in official CDF documents. Every answer includes a source citation (PDF + page number). No hallucination.

---

## Quickstart

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
ADMIN_PASSWORD=your-admin-password
```

> ⚠️ Never commit `.env` to Git. It is already listed in `.gitignore`.

For **Streamlit Cloud** deployment, add the same keys under **Settings → Secrets** when ready to deploy.

### 3. Add Documents

Place department PDFs in the `data/` folder. Name them using the department prefix convention:

```
data/
├── faq_general.pdf
├── volunteer_onboarding_guide.pdf
├── epm_admin_guide.pdf
├── hr_policies.pdf
└── finance_faq.pdf
```

The prefix before the first underscore becomes the department tag automatically (e.g. `hr_policies.pdf` → `HR`).

### 4. Build the Knowledge Base

```bash
python build_index.py
```

Run this every time you add, update, or remove a PDF. You will see a per-department chunk count confirming what was indexed.

### 5. Launch the Chatbot

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
CDF-CHATBOT/
├── .streamlit/          # UI theme settings
├── .venv/               # Virtual environment (do not commit)
├── data/                # Place all department PDFs here
├── eval/                # Evaluation scripts and question set
│   ├── eval_set.json        # 53-question evaluation set
│   ├── generate_eval_set.py # Synthetic Q&A generation from PDFs
│   ├── parse_faq.py         # Extracts real Q&A pairs from FAQ PDF
│   └── run_tuning.py        # Retrieval config comparison + scoring
├── faiss_index/         # Auto-generated vector index (do not commit)
├── .env                 # API keys and admin password (do not commit)
├── app.py               # Chatbot UI and logic
├── build_index.py       # PDF → vector index builder
└── requirements.txt     # Python dependencies
```

---

## Features

**Department Filtering**
Select a department from the sidebar to restrict answers to that department's documents only. Falls back to all departments if no match is found.

**Source Citations**
Every answer shows the source PDF filename, page number, and department tag in a collapsible "View Sources" panel.

**MMR Retrieval**
Uses Maximum Marginal Relevance (k=12, fetch_k=40) to retrieve diverse, non-redundant chunks — reduces repetition and improves answer completeness.

**Anti-Hallucination**
GPT is instructed never to guess. If the answer is not in any indexed document, the bot says so and links to the CDF Support Form.

**Partial Answers**
If only part of a question can be answered from the documents, the bot answers what it knows and flags the rest for human support.

**Smart Guardrails**
Off-topic, personal, or junk queries are identified and do not trigger the support form escalation.

**Error Handling**
Graceful handling for: empty queries, OpenAI outages, rate limits, expired API keys, and corrupted PDFs. Users always see a clean message, never a raw error.

**Admin Re-indexing UI**
Sidebar → 🔒 Admin → enter password → upload PDFs → click Upload & Rebuild Index. New documents go live immediately without restarting the app.

---

## Current Document Coverage

| Department | File | Coverage |
|---|---|---|
| FAQ | faq_general.pdf | Membership fees, waivers, payments, offboarding, verification |
| Volunteer | volunteer_onboarding_guide.pdf | Registration, onboarding, dashboard, tasks, membership |
| EPM | epm_admin_guide.pdf | Project monitoring, task tracking, team hours, metrics |

---

## Retrieval Configuration

Current production settings (tuned in Week 2):

| Parameter | Value | Notes |
|---|---|---|
| `chunk_size` | 1500 | Tokens per chunk |
| `chunk_overlap` | 150 | 10% overlap |
| `k` | 12 | Chunks retrieved per query |
| `fetch_k` | 40 | MMR candidate pool |
| Embedding model | all-MiniLM-L6-v2 | Local, no API cost |
| LLM | gpt-4o-mini | Answer generation |

---

## Evaluation Results

Scores from the 53-question eval set (22 real FAQ pairs + 14 EPM synthetic + 14 Volunteer synthetic + 3 hallucination traps):

| Metric | Week 1 Baseline | Week 2 (current) |
|---|---|---|
| Answer Accuracy | 45.0% | 89.2% |
| Grounding | — | 96.2% |
| Citation Correctness | 51.7% | 52.8% |
| I Don't Know Triggers | 100.0% | 100.0% |
| Avg Latency | 1.59s | 1.53s |

> Citation accuracy will improve significantly once department-wise PDFs are added and metadata filtering is validated.

---

## Running the Evaluation

```bash
# Generate synthetic Q&A pairs from PDFs (one-time)
python eval/generate_eval_set.py

# Extract real Q&A pairs from the FAQ PDF
python eval/parse_faq.py

# Run retrieval tuning — phase 1 (chunk size comparison)
python eval/run_tuning.py --phase 1

# Run retrieval tuning — phase 2 (k value comparison)
python eval/run_tuning.py --phase 2

# Run both phases
python eval/run_tuning.py
```

Results are saved to `eval/BASELINE.md` and `eval/tuning_results.json`.

---

## Known Limitations

- PDFs must have selectable text. Scanned or image-only PDFs will not be indexed.
- Admin UI uploads are temporary on Streamlit Cloud — container restarts wipe the `data/` folder. Migration to cloud storage (S3 / Google Drive) is planned.
- Google OAuth authentication is pending CDF admin providing Google Cloud credentials.
- Citation accuracy is currently limited by the absence of department-specific PDFs.

---

## Admin Notes

- Rebuild the index after every PDF change: `python build_index.py`
- Never commit `.env`, `.venv/`, or `faiss_index/` to the repository
- To upgrade the LLM: update the model name in `app.py` (e.g. `gpt-4o`)
- `ADMIN_PASSWORD` must be set in `.env` for the admin panel to work

---

## Contact & Support

Engineering & AI Department — file requests via the CDF Support Google Form linked within the chatbot interface.

**Repository:** https://github.com/anbunambi3108/cdf-rag-chatbot