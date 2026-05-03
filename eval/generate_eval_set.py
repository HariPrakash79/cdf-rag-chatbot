"""
generate_eval_set.py
────────────────────
Reads all PDFs in data/, samples chunks, and uses GPT to generate
question-answer pairs. Output is saved to eval/eval_set.json.

Why synthetic generation:
    We don't have a hand-curated eval set. Generating questions directly
    from the source chunks guarantees every question is answerable by the
    system — no unanswerable questions skewing accuracy scores downward.

Usage:
    python eval/generate_eval_set.py
"""

import os
import json
import random
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH         = "data/"
OUTPUT_PATH       = "eval/eval_set.json"
QUESTIONS_PER_PDF = 10    # how many Q&A pairs to generate per PDF
CHUNK_SIZE        = 1000  # use baseline chunk size for generation
CHUNK_OVERLAP     = 100
RANDOM_SEED       = 42    # reproducibility — same chunks sampled every run

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_department(filename: str) -> str:
    """Mirror the same logic as build_index.py — prefix before first underscore."""
    name = os.path.splitext(filename)[0]
    return name.split("_")[0].upper()


def generate_qa_pairs(llm: ChatOpenAI, chunk_text: str, source_file: str, n: int = 3) -> list[dict]:
    """
    Ask GPT to generate N question-answer pairs from a single chunk.

    Why we ask for JSON output:
        Structured output is easier to parse reliably than free text.
        We explicitly tell the model not to add markdown fences so we
        can call json.loads() directly without stripping backticks.

    Why we pass the source file name:
        So GPT can tag each Q&A pair with its origin — useful for
        debugging which PDFs have low coverage in the eval set.
    """
    prompt = f"""You are building an evaluation dataset for a RAG chatbot.

Given the following text from "{source_file}", generate exactly {n} question-answer pairs.

Rules:
- Each question must be fully answerable using ONLY the text provided.
- Answers must be concise (1-3 sentences).
- Questions should vary — don't just ask "what is X" for everything.
- Do NOT generate questions about page numbers, formatting, or document structure.
- Return ONLY a JSON array with no markdown fences, no extra text.

Format:
[
  {{"question": "...", "answer": "..."}},
  ...
]

Text:
\"\"\"
{chunk_text[:2000]}
\"\"\"
"""
    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Strip markdown fences if the model added them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        pairs = json.loads(raw)
        return [
            {
                "question":        p["question"],
                "expected_answer": p["answer"],
                "source_file":     source_file,
                "department":      get_department(source_file),
            }
            for p in pairs
            if "question" in p and "answer" in p
        ]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"   ⚠️  Failed to parse GPT response for {source_file}: {e}")
        return []


def main():
    os.makedirs("eval", exist_ok=True)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Why temperature=0: We want deterministic, factual Q&A pairs.
    # Creative variation would introduce ambiguity into the eval set.

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    random.seed(RANDOM_SEED)
    all_qa_pairs = []
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"❌ No PDFs found in {DATA_PATH}")
        return

    # Why we generate synthetic questions for ALL PDFs including the FAQ:
    #   FAQ pairs (from parse_faq.py) test if the chatbot gets official answers right.
    #   Synthetic pairs test a different failure mode — does the chatbot hallucinate
    #   on content that IS in the docs but wasn't explicitly listed in the FAQ?
    #   Both together give full coverage: known questions + unexpected phrasings.
    print(f"Generating synthetic questions for ALL {len(pdf_files)} PDFs")
    print("(FAQ real Q&A pairs from parse_faq.py will be merged separately)\n")

    for pdf_file in sorted(pdf_files):
        print(f"\n📄 Processing: {pdf_file}")
        loader = PyPDFLoader(os.path.join(DATA_PATH, pdf_file))
        pages  = loader.load()
        chunks = splitter.split_documents(pages)

        print(f"   {len(chunks)} chunks available")

        # Sample chunks to spread questions across the document.
        # Why random sampling instead of taking the first N chunks:
        # The first chunks are often table of contents or intro sections
        # with little answerable content. Random sampling gets better coverage.
        n_sample = min(QUESTIONS_PER_PDF, len(chunks))
        sampled  = random.sample(chunks, n_sample)

        # Generate 1-2 questions per sampled chunk to reach QUESTIONS_PER_PDF total
        questions_needed = QUESTIONS_PER_PDF
        pairs_for_pdf    = []

        for chunk in sampled:
            if len(pairs_for_pdf) >= questions_needed:
                break
            remaining = questions_needed - len(pairs_for_pdf)
            n_per_chunk = min(2, remaining)
            pairs = generate_qa_pairs(llm, chunk.page_content, pdf_file, n=n_per_chunk)
            pairs_for_pdf.extend(pairs)
            print(f"   ✅ Generated {len(pairs)} pairs from chunk")

        all_qa_pairs.extend(pairs_for_pdf)
        print(f"   📊 Total for {pdf_file}: {len(pairs_for_pdf)} questions")

    # Save eval set
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)

    print(f"\n✅ Eval set saved to {OUTPUT_PATH}")
    print(f"📊 Total questions: {len(all_qa_pairs)}")

    # Breakdown by department
    dept_counts = {}
    for pair in all_qa_pairs:
        d = pair["department"]
        dept_counts[d] = dept_counts.get(d, 0) + 1
    print("   Breakdown by department:")
    for dept, count in sorted(dept_counts.items()):
        print(f"     {dept}: {count} questions")


if __name__ == "__main__":
    main()