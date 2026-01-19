import os
from langchain_community.document_loaders import PyPDFLoader, SeleniumURLLoader, TextLoader
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

    # 1. Deep Scrape the Website (Handles JavaScript & Footers)
    print(f"🌐 Deep Scraping with Selenium: {WEB_URLS}")
    try:
        web_loader = SeleniumURLLoader(urls=WEB_URLS)
        documents.extend(web_loader.load())
    except Exception as e:
        print(f"⚠️ Web scraping failed: {e}")

    # 2. Load all files from data/ folder
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    for file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file)
        
        # Load PDFs
        if file.endswith(".pdf"):
            print(f"📄 Loading PDF: {file}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            
        # Load Plain Text Files (The "Gold Source" for facts)
        elif file.endswith(".txt"):
            print(f"📝 Loading Text Fact Sheet: {file}")
            loader = TextLoader(file_path)
            documents.extend(loader.load())

    if not documents:
        print("❌ No documents found to index.")
        return

    # 3. Split text into larger chunks (2000 chars)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=350
    )
    chunks = text_splitter.split_documents(documents)

    # 4. Metadata Enrichment: Tag Web content so the AI stays focused
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        if "http" in source:
            chunk.page_content = f"Organization: Community Dreams Foundation (CDF)\nSource: Website ({source})\n{chunk.page_content}"

    # 5. Create searchable embeddings
    print("🔗 Creating embeddings (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 6. Save to local FAISS index
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(INDEX_PATH)
    print(f"✅ Success! Context-aware index saved to '{INDEX_PATH}'")

if __name__ == "__main__":
    build()