import streamlit as st
import os
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
            st.image("cdf-logo-with-text.png", width='stretch')

st.divider()

# 5. Setup OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY! Please set it in Streamlit Secrets or your .env file.")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=True)

FORM_LINK = "https://docs.google.com/forms/d/e/1FAIpQLSeb1vE7-hXGgtqwI2mrabMB_OkFcOazp7W6oM3RaGgCegJW1w/viewform?usp=dialog"

if vector_db:
    # IMPROVED TEMPLATE: Differentiates between "Missing CDF info" and "Out of Scope"
    template = """You are the official Community Dreams Foundation (CDF) Assistant. 
    Use the provided context to answer the question. 

    - You are an expert on CDF's mission, leadership (Dion Richardson), operations, and Sebring, FL office.
    - If the user asks about the CEO, President, or Founder, refer to Dion Richardson.
    - If the user asks about the location, refer to Sebring, Florida.
    
    GUARDRAILS:
    1. If the question is UNRELATED to CDF, non-profits, or the foundation's projects (e.g., pizza, sports, general math), 
       politely state: "I am a specialized assistant for Community Dreams Foundation. I can only assist with CDF-related inquiries."
       DO NOT mention the Support Form for unrelated questions.
    
    2. If the question IS about CDF but you cannot find the specific answer in the context, 
       say: "I'm sorry, I couldn't find that specific detail in our records. Please fill out our Support Form for more help."

Context: {context}

Question: {question}
Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 8, "fetch_k": 30}
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
    if st.button("🗑️ Clear Chat History", width='stretch'):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.info("Performance: Caching is active.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Keep the button visible in history if the message contains the trigger phrase
        if "Support Form" in message["content"] and message["role"] == "assistant":
            st.link_button("📋 Open Support Form", FORM_LINK)

# 7. Chat Input
if prompt := st.chat_input("Ask about CDF (e.g., Who is the CEO?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting CDF Knowledge Base..."):
            result = qa_chain.invoke({"query": prompt})
            response = result["result"]
            sources = result["source_documents"]

            st.markdown(response)
            
            # Button only appears if the AI explicitly suggests the Support Form
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