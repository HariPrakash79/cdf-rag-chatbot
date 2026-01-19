import os
from langchain_community.document_loaders import PyPDFLoader, SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Config
DATA_PATH = "data/"
INDEX_PATH = "faiss_index"
WEB_URLS = [
    "https://cdreams.org/",
    "https://cdreams.org/team",
    "https://cdreams.org/contact"
]

def build():
    documents = []

    # 1. Deep Scrape the Website (Handles JavaScript/Footers)
    print(f"🌐 Deep Scraping with Selenium: {WEB_URLS}")
    try:
        # This acts like a real browser to capture the address at the bottom
        web_loader = SeleniumURLLoader(urls=WEB_URLS)
        documents.extend(web_loader.load())
    except Exception as e:
        print(f"⚠️ Web scraping failed: {e}")

    # 2. Load all PDFs from data/ folder
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            print(f"📄 Loading PDF: {file}")
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    if not documents:
        print("❌ No documents found to index.")
        return

    # 3. Split text into larger chunks
    # Larger chunks help keep the address linked to the company name
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=300
    )
    chunks = text_splitter.split_documents(documents)

    # 4. Metadata Enrichment (The Secret Sauce)
    # We add a prefix to every web chunk so the bot never forgets the context
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        if "http" in source:
            chunk.page_content = f"Organization: Community Dreams Foundation (CDF)\nContext: Website Content ({source})\n{chunk.page_content}"

    # 5. Create searchable embeddings
    print("🔗 Creating embeddings (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 6. Save to local FAISS index
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(INDEX_PATH)
    print(f"✅ Success! Context-aware index saved to '{INDEX_PATH}'")

if __name__ == "__main__":
    build()