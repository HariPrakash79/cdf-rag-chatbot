# 🤝 Community Dreams Foundation (CDF) Chatbot
An intelligent RAG-powered assistant designed to transform Community Dreams Foundation (CDF) PDF manuals and web content into an interactive knowledge base. It provides accurate, context-aware answers with human-in-the-loop fallback.

---

## 🌟 Project Overview
* **Semantic Search**: Uses FAISS vector storage to understand intent beyond keywords.
* **Diverse Retrieval**: Employs MMR Search (k=8) to pull information from across multiple pages.
* **Intelligent Fallback**: Provides a Support Google Form link for valid CDF questions not found in the docs.
* **Partial Fulfillment**: Answers known parts of a query while flagging missing details for support.

---

## 🚀 Recent Updates (Experimental -> Main)
We have upgraded the chatbot from a basic PDF reader to a deep-search assistant. Key technical updates include:

* **🌐 Deep Web Scraping (Selenium)**: The bot now acts like a real browser to capture information from the CDF website, including content rendered via JavaScript (like footer addresses).
* **📝 Gold Source Facts**: Added support for `.txt` fact sheets (`data/cdf_core_info.txt`) to ensure 100% accuracy for critical data like the official office address and CEO details.
* **🏷️ Metadata Contextualization**: Every piece of web data is now automatically tagged with an "Organization: CDF" prefix to prevent the AI from losing context during long conversations.
* **🧠 Improved Memory**: Increased chunk size to 2000 characters with 350-character overlap to keep complex information (like full addresses) together.

---

## 📂 Project Structure
* **app.py**: Main chatbot interface and RAG logic (updated with custom prompt templates).
* **build_index.py**: Processes PDFs, Website URLs (via Selenium), and Text fact sheets into searchable vector data.
* **data/**: Source folder for raw data.
    * `*.pdf`: Manuals and Guides.
    * `cdf_core_info.txt`: High-priority facts (Address, CEO, Contact).
* **faiss_index/**: Local vector database containing the "Brain" of the bot.
* **requirements.txt**: Updated to include `selenium`, `unstructured`, and `langchain-huggingface`.

---

## ⚙️ How to Run
1. **Environment Setup**
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. ***Configuration*** 
    Create a .env file in the root directory: OPENAI_API_KEY=your_sk_key_here

3. ***Rebuild the Knowledge Base***
     Run the indexing script to sync with the latest website and PDF content:
     python build_index.py

4. ***Launch the Bot***
    streamlit run app.py

⚠️ Limitations
Selenium Dependency: Ensure you have a stable internet connection for the initial build to scrape the web content.

PDF Formatting: PDFs must contain selectable text (the bot cannot read scanned images).