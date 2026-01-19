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
    "https://cdreams.org/contact" # Added contact page specifically
]

def build():
    documents = []

    # 1. Deep Scrape the Website (Handles JavaScript & Footers)
    print(f"🌐 Deep Scraping with Selenium: {WEB_URLS}")
    try:
        # Selenium acts like a real browser to find the hidden footer info
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
        print("❌ No documents found. Add PDFs or check the URLs.")
        return

    # 3. Split text into larger chunks
    # Increased size to 2000 so address and company stay in the same chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=250
    )
    chunks = text_splitter.split_documents(documents)

    # 4. Create searchable embeddings
    print("🔗 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 5. Save to local FAISS index
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(INDEX_PATH)
    print(f"✅ Success! Deep-indexed data saved to '{INDEX_PATH}'")

if __name__ == "__main__":
    build()