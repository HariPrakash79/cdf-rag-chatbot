🤝 Community Dreams Foundation (CDF) Chatbot
An intelligent RAG-powered assistant designed to transform Community Dreams Foundation (CDF) PDF manuals into an interactive knowledge base. It provides accurate, context-aware answers with human-in-the-loop fallback.

🌟 Project Overview
Semantic Search: Uses FAISS vector storage to understand intent beyond keywords.

Diverse Retrieval: Employs MMR Search (k=8) to pull information from across multiple pages.

Intelligent Fallback: Provides a Support Google Form link for valid CDF questions not found in the docs.

Partial Fulfillment: Answers known parts of a query while flagging missing details for support.

Smart Guardrails: Distinguishes between CDF-related inquiries and "out-of-scope" junk.

🚀 How to Run the Project
1. Environment Setup

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Configuration

Create a .env file in the root directory:
OPENAI_API_KEY=your_sk_key_here

3. Build the Knowledge Base

Place your PDFs in the data/ folder and run the indexing script to generate the faiss_index/:
python build_index.py

4. Launch the Bot

streamlit run app.py

📂 Project Structure
app.py: Main chatbot interface and RAG logic.

build_index.py: Processes PDFs into searchable vector data.

data/: Source PDF folder (Volunteer Guide, FAQ, EPM Admin Guide).

faiss_index/: Local vector database.

requirements.txt: List of necessary Python packages.

⚠️ Limitations
PDFs must contain selectable text (the bot cannot read scanned images).