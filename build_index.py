import os
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH     = "data/"
INDEX_PATH    = "faiss_index"
CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 150


def get_department(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    return name.split("_")[0].upper()


def build(chunk_size: int = CHUNK_SIZE,
          chunk_overlap: int = CHUNK_OVERLAP,
          index_path: str = INDEX_PATH):

    if not os.path.exists(DATA_PATH):
        print(f"Data folder '{DATA_PATH}' not found.")
        return

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{DATA_PATH}'.")
        return

    documents    = []
    skipped      = []
    failed       = []

    # 1. Load PDFs — skip corrupted files instead of crashing
    # Why per-file try/except instead of one big try:
    # If we wrap the whole loop, one bad PDF kills all subsequent PDFs.
    # Per-file handling lets us load everything we can and report what failed.
    for file in sorted(pdf_files):
        filepath = os.path.join(DATA_PATH, file)
        print(f"Loading: {file}")
        try:
            loader = PyPDFLoader(filepath)
            pages  = loader.load()

            if not pages:
                print(f"  WARNING: No pages extracted from {file} — skipping.")
                skipped.append(file)
                continue

            dept = get_department(file)
            for page in pages:
                page.metadata["department"] = dept

            documents.extend(pages)
            print(f"  {len(pages)} pages → department: {dept}")

        except Exception as e:
            # Log the error but continue with remaining PDFs
            print(f"  ERROR loading {file}: {type(e).__name__}: {e}")
            print(f"  Skipping {file} — ensure it has selectable text, not scanned images.")
            failed.append(file)

    if not documents:
        print("\nNo documents loaded successfully. Index was not built.")
        if failed:
            print(f"Failed files: {failed}")
        return

    # 2. Split into chunks
    print(f"\nSplitting with chunk_size={chunk_size}, overlap={chunk_overlap}...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    print(f"  {len(chunks)} chunks from {len(documents) - len(skipped) - len(failed)} PDFs")

    # 3. Create embeddings
    print("Creating embeddings (this may take a minute)...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        print(f"ERROR loading embedding model: {e}")
        return

    # 4. Build and save FAISS index
    try:
        vector_db = FAISS.from_documents(chunks, embeddings)
        vector_db.save_local(index_path)
    except Exception as e:
        print(f"ERROR building FAISS index: {e}")
        return

    # Summary
    dept_counts = {}
    for chunk in chunks:
        d = chunk.metadata.get("department", "UNKNOWN")
        dept_counts[d] = dept_counts.get(d, 0) + 1

    print(f"\nIndex saved to '{index_path}'")
    print("Chunks per department:")
    for dept, count in sorted(dept_counts.items()):
        print(f"  {dept}: {count} chunks")

    # Report any issues
    if skipped:
        print(f"\nWARNING: Skipped (no pages extracted): {skipped}")
    if failed:
        print(f"WARNING: Failed to load (corrupted or scanned): {failed}")
        print("These PDFs are NOT in the index. Fix them and re-run build_index.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size",    type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--index_path",    type=str, default=INDEX_PATH)
    args = parser.parse_args()

    build(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        index_path=args.index_path
    )