import streamlit as st
import os
import random
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 1. Load secret key
load_dotenv()

# 2. UI Configuration & Custom Styling
st.set_page_config(page_title="Community Dreams Foundation Bot", page_icon="🤝", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    .main-header { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #007BFF; font-weight: 700; margin-bottom: 0px; line-height: 1.1; }
    .sub-header { color: #28a745; font-weight: 600; margin-top: 5px; margin-bottom: 15px; }
    section[data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #eee; }
    hr { margin-top: 1rem; margin-bottom: 1rem; border: 0; border-top: 2px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

# 3. RESOURCE CACHING
@st.cache_resource
def load_resources():
    """Load the heavy models and index once and keep them in memory."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists("faiss_index"):
        vector_db = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        vector_db = None
        
    return vector_db, embeddings

vector_db, embeddings = load_resources()

# 4. Header Section
header_container = st.container()
with header_container:
    col_text, col_logo = st.columns([3, 1], vertical_alignment="center")
    with col_text:
        st.markdown('<h1 class="main-header">Welcome to Community Dreams Foundation</h1>', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">How can I assist you today?</h2>', unsafe_allow_html=True)
        st.write("I am your dedicated resource for navigating the **CDF** ecosystem.")
    with col_logo:
        if os.path.exists("cdf-logo-with-text.png"):
            st.image("cdf-logo-with-text.png", width=250)

st.divider()

# 5. Setup OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY! Please set it in Streamlit Secrets or your .env file.")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=True)

FORM_LINK = "https://docs.google.com/forms/d/e/1FAIpQLSeb1vE7-hXGgtqwI2mrabMB_OkFcOazp7W6oM3RaGgCegJW1w/viewform?usp=dialog"

if vector_db:
    # UPDATED TEMPLATE: Handles mixed queries (Valid + Invalid) intelligently
    template = """You are the official Community Dreams Foundation (CDF) Assistant. 
    Use the provided context to answer the question. 

    - You are an expert on CDF's mission, leadership (Dion Richardson), operations, and Sebring, FL office.
    - If the user asks about the CEO, President, or Founder, refer to Dion Richardson.
    - If the user asks about the location, refer to Sebring, Florida.
    
    GUIDELINES FOR ANSWERING:
    
    1. **Compound Questions (IMPORTANT)**: 
       If a user asks a question with two parts (e.g., "What is CDF and how to make pizza?"):
       - **FIRST:** Answer the CDF-related part clearly using the context.
       - **SECOND:** Politey decline the unrelated part. 
       - *Example Answer:* "The Community Dreams Foundation is a non-profit dedicated to [Mission]. However, I cannot assist with pizza recipes as I am specialized for CDF inquiries."

    2. **Strictly Unrelated**: 
       If the ENTIRE question is unrelated (e.g., "How to fix a car"), refuse it:
       "I am a specialized assistant for Community Dreams Foundation. I can only assist with CDF-related inquiries."
    
    3. **Missing Info**: 
       If the question is about CDF but the answer is not in the context, say:
       "I'm sorry, I couldn't find that specific detail in our records. Please fill out our Support Form for more help."

    Context: {context}

    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(
            search_type="mmr", 
            # Increased k to 10 for better mission-statement retrieval
            search_kwargs={"k": 10, "fetch_k": 40} 
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
else:
    st.warning("⚠️ Knowledge base not found. Please run 'python build_index.py' first.")
    st.stop()

# 6. Interface & Sidebar
with st.sidebar:
    st.title("Settings")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.suggestion_idx = random.randint(0, 5)
        st.rerun()
    st.markdown("---")
    st.info("Performance: Caching is active.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Refined Suggestions: Uses full organization name for better vector matching
suggestions = [
    "Who is Dion Richardson?",
    "Where is the CDF office located?",
    "Tell me about the Community Dreams Foundation mission.",
    "How do I contact Dion Richardson?",
    "What is the address in Sebring, Florida?",
    "What does CDF do?"
]

if "suggestion_idx" not in st.session_state:
    st.session_state.suggestion_idx = 0

# 7. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "Support Form" in message["content"] and message["role"] == "assistant":
            st.link_button("📋 Open Support Form", FORM_LINK)

# 8. Chat Input with Focus and Suggestion Rotation Fix
current_placeholder = f"Ask about CDF (e.g., {suggestions[st.session_state.suggestion_idx]})"

# The key="chat_input" ensures the browser keeps focus in this box after st.rerun()
if prompt := st.chat_input(current_placeholder, key="chat_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.suggestion_idx = (st.session_state.suggestion_idx + 1) % len(suggestions)
    st.rerun()

# 9. Response Logic
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("Consulting CDF Knowledge Base..."):
            result = qa_chain.invoke({"query": user_prompt})
            response = result["result"]
            sources = result["source_documents"]

            st.markdown(response)
            
            if "Support Form" in response:
                st.link_button("📋 Open Support Form", FORM_LINK, type="primary")

            if sources and "specialized assistant" not in response:
                with st.expander("📚 View Reference Sources"):
                    for doc in sources:
                        raw_source = doc.metadata.get('source', 'Unknown')
                        if raw_source.startswith('http'):
                            st.write(f"- **Web:** [{raw_source}]({raw_source})")
                        else:
                            source_name = os.path.basename(raw_source)
                            page_num = doc.metadata.get('page', 0) + 1
                            st.write(f"- **PDF:** {source_name} (Page {page_num})")

            st.session_state.messages.append({"role": "assistant", "content": response})