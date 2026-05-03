"""
parse_faq.py
────────────
Reads the FAQ PDF and extracts all Q&A pairs into eval/eval_set.json.

Why this is better than synthetic generation for FAQ questions:
    Synthetic generation invents questions GPT thinks users might ask.
    The FAQ contains questions real users actually asked — ground truth
    doesn't get more reliable than that. The expected answers are also
    the official CDF answers, so accuracy scoring is meaningful.

Why we use GPT to extract instead of regex:
    The FAQ PDF has inconsistent numbering (5, 6, 9, 10, 25 are missing —
    likely on pages not included in this version). Regex would break on
    gaps. GPT handles arbitrary formatting gracefully.

After running this script, run generate_eval_set.py to fill in questions
from the other PDFs (volunteer guide, EPM admin guide, etc.) that the
FAQ doesn't cover.

Usage:
    python eval/parse_faq.py
"""

import os
import json
from dotenv import load_dotenv
import pdfplumber
from langchain_openai import ChatOpenAI

load_dotenv()

FAQ_PDF_PATH = "data/Frequently_Asked_Questions.pdf"
OUTPUT_PATH  = "eval/eval_set.json"
SOURCE_FILE  = "Frequently_Asked_Questions.pdf"
DEPARTMENT   = "FAQ"


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from the FAQ PDF using pdfplumber.

    Why pdfplumber over PyPDFLoader:
        pdfplumber preserves layout better for text-heavy docs with
        numbered lists, making it easier for GPT to identify Q&A boundaries.
    """
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n\n".join(full_text)


def extract_qa_pairs(llm: ChatOpenAI, text: str) -> list[dict]:
    """
    Ask GPT to extract all Q&A pairs from the raw FAQ text.

    Why a single GPT call over the full text:
        The FAQ is short enough (~4 pages) to fit in one prompt.
        One call is cheaper and avoids inconsistencies from chunking.
        We explicitly ask for JSON-only output to make parsing reliable.
    """
    prompt = f"""You are extracting Q&A pairs from a FAQ document.

Read the following FAQ text and extract every question-answer pair.

Rules:
- Extract ALL questions and their complete answers — do not skip any.
- Keep the original question wording exactly as written.
- Keep the full answer text — do not truncate.
- If a question number is missing (e.g., jumps from 4 to 7), that's fine — just extract what's there.
- Return ONLY a JSON array with no markdown fences, no extra text.

Format:
[
  {{"question": "...", "answer": "..."}},
  ...
]

FAQ Text:
\"\"\"
{text}
\"\"\"
"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Strip markdown fences if model added them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    pairs = json.loads(raw)
    return pairs


def main():
    os.makedirs("eval", exist_ok=True)

    # Validate FAQ PDF exists
    if not os.path.exists(FAQ_PDF_PATH):
        print(f"FAQ PDF not found at {FAQ_PDF_PATH}")
        print(f"Place the FAQ PDF at: {FAQ_PDF_PATH}")
        return

    print(f"Reading FAQ PDF: {FAQ_PDF_PATH}")
    text = extract_text_from_pdf(FAQ_PDF_PATH)
    print(f"Extracted {len(text)} characters from FAQ PDF")

    print("Extracting Q&A pairs with GPT...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # temperature=0: extraction should be deterministic — no creativity needed

    raw_pairs = extract_qa_pairs(llm, text)
    print(f"Found {len(raw_pairs)} Q&A pairs")

    # Format into eval set structure
    # Why we tag source_file and department:
    #   run_tuning.py checks if the source doc retrieved by the RAG system
    #   matches source_file to compute citation accuracy. department is used
    #   for grouping in the breakdown summary.
    eval_entries = [
        {
            "question":        pair["question"],
            "expected_answer": pair["answer"],
            "source_file":     SOURCE_FILE,
            "department":      DEPARTMENT,
        }
        for pair in raw_pairs
        if pair.get("question") and pair.get("answer")
    ]

    # Load existing eval set if it exists (merge, don't overwrite)
    existing = []
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            existing = json.load(f)
        # Remove any old FAQ entries to avoid duplicates on re-run
        existing = [e for e in existing if e.get("source_file") != SOURCE_FILE]
        print(f"Loaded {len(existing)} existing non-FAQ entries from {OUTPUT_PATH}")

    final = existing + eval_entries

    with open(OUTPUT_PATH, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nEval set saved to {OUTPUT_PATH}")
    print(f"  FAQ questions:       {len(eval_entries)}")
    print(f"  Non-FAQ (existing):  {len(existing)}")
    print(f"  Total:               {len(final)}")

    # Preview first 3
    print("\nSample entries:")
    for entry in eval_entries[:3]:
        print(f"  Q: {entry['question'][:80]}...")
        print(f"  A: {entry['expected_answer'][:80]}...")
        print()


if __name__ == "__main__":
    main()