import streamlit as st
import os
import time
import openai
import hashlib
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Import build() directly — avoids spawning a subprocess which would
# reload the 90MB embedding model from scratch. Direct import reuses
# the same Python process and cached resources.
from build_index import build as rebuild_index

# ── 1. Load secrets ──────────────────────────────────────────────────────────
load_dotenv()

# ── 2. Page config — must be first Streamlit call ────────────────────────────
st.set_page_config(page_title="CDF Onboarding Bot", page_icon="🤝")

# ── 3. Validate API key early ─────────────────────────────────────────────────
# Why before loading the embedding model: fail fast with a clear message
# rather than wasting time loading a 90MB model with no API key.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 Missing OPENAI_API_KEY. Add it to your .env file or Streamlit Cloud secrets.")
    st.stop()

# ── 4. Session state init ─────────────────────────────────────────────────────
# Why initialise here: Streamlit reruns the entire script on every interaction.
# We need these flags to persist across reruns within the same session.
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 5. Cached resource loaders ────────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"❌ Failed to load embedding model: {e}")
        st.stop()

@st.cache_resource
def load_vector_db(_embeddings):
    try:
        return FAISS.load_local(
            "faiss_index",
            _embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(
            f"❌ Failed to load knowledge base: {e}\n\n"
            "The index may be corrupted. Run `python build_index.py` to rebuild it."
        )
        st.stop()

@st.cache_data
def get_available_departments(_vector_db) -> list[str]:
    """
    Read unique department values from FAISS index metadata.
    Dynamic so new PDFs auto-appear in the dropdown after re-indexing.
    """
    try:
        departments = set()
        for doc_id in _vector_db.index_to_docstore_id.values():
            doc = _vector_db.docstore.search(doc_id)
            if hasattr(doc, "metadata") and "department" in doc.metadata:
                departments.add(doc.metadata["department"])
        return sorted(departments)
    except Exception:
        return []


# ── 6. Load knowledge base ────────────────────────────────────────────────────
if not os.path.exists("faiss_index"):
    st.warning("⚠️ Knowledge base not found. Please run `python build_index.py` first.")
    st.stop()

embeddings            = load_embeddings()
vector_db             = load_vector_db(embeddings)
available_departments = get_available_departments(vector_db)

# ── 7. LLM ────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# ── 8. Constants ──────────────────────────────────────────────────────────────
FORM_LINK        = "https://docs.google.com/forms/d/e/1FAIpQLSeb1vE7-hXGgtqwI2mrabMB_OkFcOazp7W6oM3RaGgCegJW1w/viewform?usp=dialog"
MAX_QUERY_LENGTH = 500
DATA_PATH        = "data/"

# ── 9. Prompt template ────────────────────────────────────────────────────────
template = """You are a CDF Document Expert. Use the provided context to answer the user's question accurately.

    - If the user's question has multiple parts, answer every part you can find in the context.
    - If you find some information but not all (e.g., you find registration steps but not core values), provide the information you found and then add: 
      "Note: I couldn't find information regarding [the missing part] in our documents. For that specific inquiry, please fill out our Support Form."
    - If the entire question is UNRELATED to CDF (personal questions, jokes, or general trivia), politely state that you only assist with CDF documentation and do NOT provide the form link.
    - If the question IS about CDF but you find NOTHING at all, use this specific fallback: 
      "I'm sorry, I couldn't find that information in our current documentation. Please fill out our Support Form and someone from our team will respond to you via email."

Context: {context}

Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)


# ── 10. Helpers ───────────────────────────────────────────────────────────────
def build_qa_chain(department_filter: str | None):
    search_kwargs = {"k": 12, "fetch_k": 40}
    if department_filter:
        search_kwargs["filter"] = {"department": department_filter}
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )


def validate_query(query: str) -> tuple[bool, str]:
    stripped = query.strip()
    if not stripped:
        return False, "Please enter a question before submitting."
    if len(stripped) < 3:
        return False, "Your question is too short. Please be more specific."
    if len(stripped) > MAX_QUERY_LENGTH:
        return False, f"Your question is too long ({len(stripped)} chars). Please keep it under {MAX_QUERY_LENGTH} characters."
    return True, ""


def run_qa_with_error_handling(query: str, department_filter: str | None) -> tuple[str, list]:
    try:
        qa_chain = build_qa_chain(department_filter)
        result   = qa_chain.invoke({"query": query})
        return result["result"], result["source_documents"]
    except openai.RateLimitError:
        return "⚠️ We've hit the OpenAI API rate limit. Please wait a moment and try again.", []
    except openai.APIConnectionError:
        return "⚠️ Couldn't connect to OpenAI. Please check your internet connection and try again.", []
    except openai.APIStatusError as e:
        if e.status_code >= 500:
            return ("⚠️ OpenAI is experiencing an outage. Please try again in a few minutes. "
                    "Check https://status.openai.com", [])
        return f"⚠️ OpenAI returned an error (status {e.status_code}). Please try again.", []
    except openai.AuthenticationError:
        return "🔑 The OpenAI API key is invalid or expired. Please contact the admin to update it.", []
    except Exception as e:
        print(f"[ERROR] Unexpected QA error: {type(e).__name__}: {e}")
        return ("⚠️ Something went wrong. Please try again. "
                "If the problem persists, use the Support Form below.", [])


def check_admin_password(entered: str) -> bool:
    """
    Compare entered password against ADMIN_PASSWORD from env.

    Why hash comparison instead of direct string compare:
    Comparing hashes prevents timing attacks — an attacker can't
    determine the password length or content from response time differences.
    Why not store password in code: hardcoded passwords end up in Git history.
    """
    admin_pw = os.getenv("ADMIN_PASSWORD", "")
    if not admin_pw:
        return False
    entered_hash = hashlib.sha256(entered.encode()).hexdigest()
    stored_hash  = hashlib.sha256(admin_pw.encode()).hexdigest()
    return entered_hash == stored_hash


def save_uploaded_pdfs(uploaded_files) -> tuple[list[str], list[str]]:
    """
    Save uploaded PDF files to the data/ folder.
    Returns (saved_filenames, failed_filenames).

    Why validate PDF magic bytes:
    st.file_uploader with type=["pdf"] checks the extension but not the
    actual file content. A renamed .exe would pass the extension check.
    Checking the first 4 bytes (%PDF) ensures it's actually a PDF.
    """
    os.makedirs(DATA_PATH, exist_ok=True)
    saved  = []
    failed = []

    for f in uploaded_files:
        try:
            content = f.read()

            # Validate PDF magic bytes
            if not content.startswith(b"%PDF"):
                failed.append(f.name)
                continue

            filepath = os.path.join(DATA_PATH, f.name)
            with open(filepath, "wb") as out:
                out.write(content)
            saved.append(f.name)

        except Exception as e:
            print(f"[ERROR] Failed to save {f.name}: {e}")
            failed.append(f.name)

    return saved, failed


# ── 11. Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    # Department filter
    dept_options   = ["All Departments"] + available_departments
    selected_label = st.selectbox(
        "🗂️ Filter by Department",
        options=dept_options,
        help="Narrow answers to a specific department's documents."
    )
    selected_dept = None if selected_label == "All Departments" else selected_label

    if selected_dept:
        st.info(f"Searching: **{selected_dept}** documents only")
    else:
        st.info("Searching: All departments")

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Tip: If you have multiple questions, I can answer them all at once if they are in the docs!")

    # ── Admin Panel ───────────────────────────────────────────────────────────
    # Hidden at the bottom — only visible after correct password.
    # Why password in session_state: Streamlit reruns the script on every
    # interaction. Without session_state the admin would be logged out
    # every time they click anything.
    st.divider()
    with st.expander("🔒 Admin"):
        if not st.session_state.admin_authenticated:
            pw = st.text_input(
                "Password",
                type="password",
                key="admin_pw_input",
                label_visibility="collapsed",
                placeholder="Admin password"
            )
            if st.button("Unlock", key="admin_unlock"):
                if check_admin_password(pw):
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")

        else:
            # ── Admin is authenticated ────────────────────────────────────────
            st.success("✅ Admin access granted")

            st.markdown("**Upload Department PDFs**")
            st.caption(
                "Name files as `<department>_<anything>.pdf` "
                "e.g. `hr_policies.pdf`, `finance_faq.pdf`"
            )

            uploaded = st.file_uploader(
                "Choose PDF files",
                type=["pdf"],
                accept_multiple_files=True,
                key="admin_uploader",
                label_visibility="collapsed"
            )

            if uploaded:
                st.markdown(f"**{len(uploaded)} file(s) selected:**")
                for f in uploaded:
                    st.write(f"📄 {f.name}")

                if st.button("📤 Upload & Rebuild Index", key="admin_rebuild"):
                    # Step 1 — Save PDFs
                    with st.spinner("Saving PDFs to data/ folder..."):
                        saved, failed = save_uploaded_pdfs(uploaded)

                    if failed:
                        st.warning(f"⚠️ Could not save: {', '.join(failed)}")
                    if not saved:
                        st.error("No valid PDFs were saved. Rebuild cancelled.")
                    else:
                        st.success(f"✅ Saved: {', '.join(saved)}")

                        # Step 2 — Rebuild index
                        with st.spinner("Rebuilding knowledge base... this may take a minute."):
                            try:
                                # Clear cached resources so the new index is
                                # loaded on next user query — not the old one.
                                st.cache_resource.clear()
                                st.cache_data.clear()

                                rebuild_index()

                                st.success("✅ Knowledge base rebuilt successfully!")
                                st.info("The new documents are now live. Refresh the page to load the updated index.")

                            except Exception as e:
                                st.error(
                                    f"❌ Rebuild failed: {e}\n\n"
                                    "The old knowledge base is still active. "
                                    "Check the terminal for details."
                                )

            # Logout button
            if st.button("🔓 Lock Admin", key="admin_logout"):
                st.session_state.admin_authenticated = False
                st.rerun()


# ── 12. Page header ───────────────────────────────────────────────────────────
st.markdown("<h1 style='color: #007BFF;'>🤝 CDF Volunteer Onboarding Bot</h1>", unsafe_allow_html=True)
st.write("Welcome! I'm here to help new volunteers navigate the **Community of Developers (CDF)**.")

# ── 13. Chat interface ────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "Support Form" in message["content"] and message["role"] == "assistant":
            st.link_button("📋 Open Support Form", FORM_LINK)

if prompt := st.chat_input("Ask about CDF volunteering..."):

    is_valid, error_msg = validate_query(prompt)
    if not is_valid:
        st.warning(f"⚠️ {error_msg}")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching CDF guides..."):
                response, sources = run_qa_with_error_handling(prompt, selected_dept)

            st.markdown(response)

            if "Support Form" in response:
                st.link_button("📋 Open Support Form", FORM_LINK)

            if sources and "only assist with CDF" not in response:
                with st.expander("📚 View Sources"):
                    seen = set()
                    for doc in sources:
                        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                        page_num    = doc.metadata.get("page", 0) + 1
                        dept        = doc.metadata.get("department", "")
                        key         = (source_name, page_num)
                        if key not in seen:
                            seen.add(key)
                            dept_tag = f" · `{dept}`" if dept else ""
                            st.write(f"- **{source_name}** (Page {page_num}){dept_tag}")

        st.session_state.messages.append({"role": "assistant", "content": response})