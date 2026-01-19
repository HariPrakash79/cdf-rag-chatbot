import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Config
DATA_PATH = "data/"
INDEX_PATH = "faiss_index"
# Official CDF URLs to scrape
WEB_URLS = [
    "https://cdreams.org/",
    "https://cdreams.org/team" # Scrapes CEO and leadership info
]

def build():
    documents = []

    # 1. Load data from the CDF Website
    print(f"🌐 Scraping: {WEB_URLS}")
    try:
        web_loader = WebBaseLoader(WEB_URLS)
        documents.extend(web_loader.load())
    except Exception as e:
        print(f"⚠️ Web scraping failed (check your internet): {e}")

    # 2. Load all PDFs from data/ folder
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"📁 Created '{DATA_PATH}' folder. Put your PDFs here!")
    
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            print(f"📄 Loading PDF: {file}")
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    if not documents:
        print("❌ No documents found to index. Add PDFs or check the URLs.")
        return

    # 3. Split text into manageable chunks
    # We use a slightly smaller overlap to keep web text concise
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # 4. Create searchable embeddings
    print("🔗 Creating embeddings (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 5. Save to local FAISS index
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(INDEX_PATH)
    print(f"✅ Success! Search index (Web + PDF) saved to '{INDEX_PATH}'")

if __name__ == "__main__":
    build()