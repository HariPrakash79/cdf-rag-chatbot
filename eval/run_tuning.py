"""
run_tuning.py
─────────────
Tests multiple retrieval configs against the eval set and outputs
a comparison table so you can pick the best config for production.

Two-phase tuning strategy:
    Phase 1 — Fix k=8, vary chunk_size: 500 / 1000 / 1500
              Answers: does smaller/larger context improve accuracy?
    Phase 2 — Fix best chunk_size, vary k: 4 / 12
              (k=8 result already exists from Phase 1)
              Answers: does more/fewer retrieved chunks help?

Why not test all 9 combinations:
    5 runs × N questions × 2 LLM calls (retrieval + judge) is already
    significant API cost. Two-phase isolates each variable cleanly.

Scoring (LLM-as-judge):
    Each answer is scored 0.0–1.0 on:
      - accuracy:   does the answer correctly address the question?
      - grounding:  is the answer based on the retrieved docs (not hallucinated)?
    Final score = average of both dimensions.

Usage:
    # First generate the eval set:
    python eval/generate_eval_set.py

    # Then run tuning:
    python eval/run_tuning.py
    
    # Or run a single phase:
    python eval/run_tuning.py --phase 1
    python eval/run_tuning.py --phase 2
"""

import os
import json
import time
import argparse
import tempfile
import shutil
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH      = "data/"
EVAL_SET_PATH  = "eval/eval_set.json"
RESULTS_PATH   = "eval/tuning_results.json"
BASELINE_PATH  = "eval/BASELINE.md"

# ── Prompt (same as production app.py — must be identical for fair comparison) 
PROMPT_TEMPLATE = """You are a CDF Document Expert. Use the provided context to answer the user's question accurately.

    - If the user's question has multiple parts, answer every part you can find in the context.
    - If you find some information but not all, provide what you found and flag the rest.
    - If the entire question is UNRELATED to CDF, politely state that you only assist with CDF documentation.
    - If the question IS about CDF but you find NOTHING at all, say you couldn't find that information.

Context: {context}

Question: {question}
Helpful Answer:"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE
)


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class TuningConfig:
    chunk_size:   int
    chunk_overlap: int
    k:            int
    fetch_k:      int
    label:        str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"chunk={self.chunk_size}_k={self.k}"


@dataclass
class EvalResult:
    config:             TuningConfig
    accuracy_scores:    list[float] = field(default_factory=list)
    grounding_scores:   list[float] = field(default_factory=list)
    latencies:          list[float] = field(default_factory=list)
    citation_hits:      int = 0
    total_questions:    int = 0

    @property
    def avg_accuracy(self)  -> float:
        return round(sum(self.accuracy_scores)  / len(self.accuracy_scores),  3) if self.accuracy_scores  else 0.0

    @property
    def avg_grounding(self) -> float:
        return round(sum(self.grounding_scores) / len(self.grounding_scores), 3) if self.grounding_scores else 0.0

    @property
    def avg_latency(self)   -> float:
        return round(sum(self.latencies)        / len(self.latencies),        3) if self.latencies        else 0.0

    @property
    def citation_accuracy(self) -> float:
        return round(self.citation_hits / self.total_questions, 3) if self.total_questions else 0.0


# ── Build temporary index ─────────────────────────────────────────────────────
def build_temp_index(config: TuningConfig, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Build a FAISS index with the given chunk config in a temp directory.

    Why temp directory:
        We don't want to overwrite the production faiss_index/ during tuning.
        Each config gets its own isolated index that's discarded after scoring.
    """
    documents = []
    for file in os.listdir(DATA_PATH):
        if not file.endswith(".pdf"):
            continue
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        pages  = loader.load()
        dept   = os.path.splitext(file)[0].split("_")[0].upper()
        for page in pages:
            page.metadata["department"] = dept
        documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)


# ── LLM-as-judge ─────────────────────────────────────────────────────────────
def judge_answer(
    llm:             ChatOpenAI,
    question:        str,
    expected_answer: str,
    actual_answer:   str,
    source_docs:     list,
    source_file:     str,
) -> tuple[float, float, bool]:
    """
    Ask GPT to score the RAG answer on accuracy and grounding.

    Returns:
        accuracy  (0.0–1.0) — does the answer correctly address the question?
        grounding (0.0–1.0) — is the answer based on docs, not hallucinated?
        citation_hit (bool) — did any source doc match the expected source file?

    Why LLM-as-judge instead of keyword matching:
        Keyword matching penalizes correct paraphrases. For example,
        "The fee is $15/month" and "Monthly cost is fifteen dollars" are
        both correct but share no keywords. GPT understands semantic equivalence.

    Why we separate accuracy and grounding:
        A model can be accurate (right answer) but hallucinated (not from docs).
        We want to catch cases where the model gets lucky with parametric memory
        instead of actually retrieving the right chunk.
    """
    prompt = f"""You are an evaluator for a RAG chatbot. Score the following answer.

Question: {question}
Expected Answer: {expected_answer}
Actual Answer: {actual_answer}

Score the actual answer on two dimensions (0.0 to 1.0 each):
1. accuracy:   Does the actual answer correctly address the question? (1.0 = fully correct, 0.5 = partially correct, 0.0 = wrong or irrelevant)
2. grounding:  Is the actual answer based on retrieved documents rather than hallucinated? (1.0 = clearly grounded, 0.0 = clearly hallucinated or made up)

Return ONLY a JSON object with no markdown fences:
{{"accuracy": <float>, "grounding": <float>, "reason": "<one sentence>"}}"""

    response = llm.invoke(prompt)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        scores = json.loads(raw)
        accuracy  = float(scores.get("accuracy",  0.0))
        grounding = float(scores.get("grounding", 0.0))
    except (json.JSONDecodeError, ValueError):
        accuracy  = 0.0
        grounding = 0.0

    # Citation hit: does any retrieved source doc match the expected source file?
    citation_hit = any(
        os.path.basename(doc.metadata.get("source", "")) == source_file
        for doc in source_docs
    )

    return accuracy, grounding, citation_hit


# ── Evaluate one config ───────────────────────────────────────────────────────
def evaluate_config(
    config:    TuningConfig,
    eval_set:  list[dict],
    embeddings: HuggingFaceEmbeddings,
    llm:       ChatOpenAI,
    judge_llm: ChatOpenAI,
) -> EvalResult:
    """
    Build a temp index with the config, run every eval question through it,
    score with LLM-as-judge, return aggregated EvalResult.
    """
    print(f"\n{'─'*60}")
    print(f"🔧 Testing config: {config.label}")
    print(f"   chunk_size={config.chunk_size}, overlap={config.chunk_overlap}, k={config.k}, fetch_k={config.fetch_k}")
    print(f"{'─'*60}")

    # Build temp index
    print("   Building index...")
    vector_db = build_temp_index(config, embeddings)

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.k, "fetch_k": config.fetch_k}
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    result = EvalResult(config=config, total_questions=len(eval_set))

    for i, item in enumerate(eval_set, 1):
        question        = item["question"]
        expected_answer = item["expected_answer"]
        source_file     = item["source_file"]

        print(f"   [{i:02d}/{len(eval_set)}] {question[:70]}...")

        # Run the question through the chain and measure latency
        start = time.time()
        try:
            output       = qa_chain.invoke({"query": question})
            actual_answer = output["result"]
            source_docs   = output["source_documents"]
        except Exception as e:
            print(f"         ❌ Chain error: {e}")
            actual_answer = ""
            source_docs   = []
        latency = round(time.time() - start, 3)

        # Score with LLM-as-judge
        accuracy, grounding, citation_hit = judge_answer(
            judge_llm, question, expected_answer,
            actual_answer, source_docs, source_file
        )

        result.accuracy_scores.append(accuracy)
        result.grounding_scores.append(grounding)
        result.latencies.append(latency)
        if citation_hit:
            result.citation_hits += 1

        print(f"         acc={accuracy:.2f}  grounding={grounding:.2f}  citation={'✓' if citation_hit else '✗'}  latency={latency}s")

    return result


# ── Report ────────────────────────────────────────────────────────────────────
def print_comparison_table(results: list[EvalResult]):
    """Print a formatted comparison table and save to BASELINE.md."""

    header = (
        f"\n{'Config':<22} {'Accuracy':>10} {'Grounding':>10} "
        f"{'Citation':>10} {'Latency':>10}"
    )
    divider = "─" * 66
    rows = []

    for r in results:
        row = (
            f"{r.config.label:<22} "
            f"{r.avg_accuracy:>9.1%} "
            f"{r.avg_grounding:>9.1%} "
            f"{r.citation_accuracy:>9.1%} "
            f"{r.avg_latency:>8.2f}s"
        )
        rows.append(row)

    print(header)
    print(divider)
    for row in rows:
        print(row)
    print(divider)

    # Find best config
    best = max(results, key=lambda r: r.avg_accuracy)
    print(f"\n🏆 Best config: {best.config.label}")
    print(f"   Accuracy:  {best.avg_accuracy:.1%}")
    print(f"   Grounding: {best.avg_grounding:.1%}")
    print(f"   Citation:  {best.citation_accuracy:.1%}")
    print(f"   Latency:   {best.avg_latency:.2f}s")
    print(f"\n   → Update build_index.py: chunk_size={best.config.chunk_size}")
    print(f"   → Update app.py retriever: k={best.config.k}, fetch_k={best.config.fetch_k}")

    # Save markdown report
    md_lines = [
        "# Retrieval Tuning Results\n",
        f"| Config | Accuracy | Grounding | Citation | Latency |",
        f"|--------|----------|-----------|----------|---------|",
    ]
    for r in results:
        md_lines.append(
            f"| {r.config.label} | {r.avg_accuracy:.1%} | "
            f"{r.avg_grounding:.1%} | {r.citation_accuracy:.1%} | {r.avg_latency:.2f}s |"
        )
    md_lines.append(f"\n**Best config:** `{best.config.label}`")
    md_lines.append(
        f"\n**Recommended changes:**\n"
        f"- `build_index.py`: `chunk_size={best.config.chunk_size}`\n"
        f"- `app.py` retriever: `k={best.config.k}`, `fetch_k={best.config.fetch_k}`"
    )

    with open(BASELINE_PATH, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\n📄 Full report saved to {BASELINE_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                        help="Run only phase 1 (chunk size) or phase 2 (k values). Default: both.")
    args = parser.parse_args()

    # Load eval set
    if not os.path.exists(EVAL_SET_PATH):
        print(f"❌ Eval set not found at {EVAL_SET_PATH}")
        print("   Run: python eval/generate_eval_set.py first")
        return

    with open(EVAL_SET_PATH) as f:
        eval_set = json.load(f)
    print(f"✅ Loaded {len(eval_set)} eval questions")

    # ── Phase 1 configs: fix k=8, vary chunk_size ──────────────────────────
    phase1_configs = [
        TuningConfig(chunk_size=500,  chunk_overlap=50,  k=8, fetch_k=30, label="chunk=500_k=8"),
        TuningConfig(chunk_size=1000, chunk_overlap=100, k=8, fetch_k=30, label="chunk=1000_k=8 (baseline)"),
        TuningConfig(chunk_size=1500, chunk_overlap=150, k=8, fetch_k=30, label="chunk=1500_k=8"),
    ]

    # ── Phase 2 configs: fix best chunk (determined after phase 1), vary k ──
    # We insert the best chunk_size from phase 1 here.
    # k=8 is already tested in phase 1 so we only test 4 and 12.
    # fetch_k is always ~3-4x k to give MMR enough candidates to diversify.
    phase2_base_chunk = 1500  # updated after phase 1 runs
    phase2_configs = [
        TuningConfig(chunk_size=phase2_base_chunk, chunk_overlap=100, k=4,  fetch_k=15, label=f"chunk={phase2_base_chunk}_k=4"),
        TuningConfig(chunk_size=phase2_base_chunk, chunk_overlap=100, k=12, fetch_k=40, label=f"chunk={phase2_base_chunk}_k=12"),
    ]

    # Load models once — reused across all configs
    # Why load embeddings once: the embedding model is ~90MB and takes ~5s to load.
    # Reloading for each config would add minutes to the total tuning time.
    print("\n🔗 Loading embedding model (once for all configs)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    llm       = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Why temperature=0 for judge: scoring should be deterministic.
    # Same answer should always get the same score.

    all_results = []

    # ── Run phase 1 ───────────────────────────────────────────────────────
    if args.phase in (None, 1):
        print("\n" + "═"*60)
        print("PHASE 1: Chunk Size Comparison (k=8 fixed)")
        print("═"*60)
        phase1_results = []
        for config in phase1_configs:
            result = evaluate_config(config, eval_set, embeddings, llm, judge_llm)
            phase1_results.append(result)
            all_results.append(result)

        # Determine best chunk size for phase 2
        best_p1 = max(phase1_results, key=lambda r: r.avg_accuracy)
        phase2_base_chunk = best_p1.config.chunk_size
        print(f"\n✅ Phase 1 winner: chunk_size={phase2_base_chunk}")

        # Update phase 2 configs with the winning chunk size
        phase2_configs = [
            TuningConfig(chunk_size=phase2_base_chunk, chunk_overlap=max(50, phase2_base_chunk//10),
                         k=4,  fetch_k=15, label=f"chunk={phase2_base_chunk}_k=4"),
            TuningConfig(chunk_size=phase2_base_chunk, chunk_overlap=max(50, phase2_base_chunk//10),
                         k=12, fetch_k=40, label=f"chunk={phase2_base_chunk}_k=12"),
        ]

    # ── Run phase 2 ───────────────────────────────────────────────────────
    if args.phase in (None, 2):
        print("\n" + "═"*60)
        print(f"PHASE 2: k Value Comparison (chunk_size={phase2_base_chunk} fixed)")
        print("═"*60)
        for config in phase2_configs:
            result = evaluate_config(config, eval_set, embeddings, llm, judge_llm)
            all_results.append(result)

    # ── Output comparison table ────────────────────────────────────────────
    if all_results:
        print_comparison_table(all_results)

        # Save raw results as JSON for further analysis
        with open(RESULTS_PATH, "w") as f:
            json.dump(
                [
                    {
                        "config":           asdict(r.config),
                        "avg_accuracy":     r.avg_accuracy,
                        "avg_grounding":    r.avg_grounding,
                        "avg_latency":      r.avg_latency,
                        "citation_accuracy": r.citation_accuracy,
                        "total_questions":  r.total_questions,
                    }
                    for r in all_results
                ],
                f, indent=2
            )
        print(f"📊 Raw results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()