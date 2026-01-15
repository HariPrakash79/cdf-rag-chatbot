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

# Custom CSS for a professional, versatile app look
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #007BFF;
        font-weight: 700;
        margin-bottom: 0px;
        line-height: 1.1;
    }
    .sub-header {
        color: #28a745;
        font-weight: 600;
        margin-top: 5px;
        margin-bottom: 15px;
    }
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #eee;
    }
    hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
        border: 0;
        border-top: 2px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

# Header Section - Generalized Salutation
header_container = st.container()
with header_container:
    col_text, col_logo = st.columns([3, 1], vertical_alignment="center")
    
    with col_text:
        st.markdown('<h1 class="main-header">Welcome to Community Dreams Foundation</h1>', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">How can I assist you today?</h2>', unsafe_allow_html=True)
        st.write("I am your dedicated resource for navigating the **CDF** ecosystem. Ask me anything about our programs, documentation, or initiatives.")
    
    with col_logo:
        st.image("cdf-logo-with-text.png", use_container_width=True)

st.divider()

# 3. Setup OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY in your .env file!")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 4. Load Knowledge Base
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Constant for the form link
FORM_LINK = "https://docs.google.com/forms/d/e/1FAIpQLSeb1vE7-hXGgtqwI2mrabMB_OkFcOazp7W6oM3RaGgCegJW1w/viewform?usp=dialog"

if os.path.exists("faiss_index"):
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Updated Prompt Template - Generalized for all users
    template = """You are a CDF Document Expert. Use the provided context to answer the user's question accurately.

    - If the user's question has multiple parts, answer every part you can find in the context.
    - If you find some information but not all, provide what you found and then add: 
      "Note: I couldn't find information regarding [the missing part] in our documents. For that specific inquiry, please fill out our Support Form."
    - If the question is UNRELATED to CDF, politely state that you only assist with CDF-related information and do NOT provide the form link.
    - If the question IS about CDF but you find NOTHING at all, use this specific fallback: 
      "I'm sorry, I couldn't find that information in our current documentation. Please fill out our Support Form and someone from our team will respond to you via email."

Context: {context}

Question: {question}
Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

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
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 Resource Links")
    st.info("Ask about programs, technical documentation, or contact information.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "Support Form" in message["content"] and message["role"] == "assistant":
            st.link_button("📋 Open Support Form", FORM_LINK)

# Chat Input
if prompt := st.chat_input("Ask a question about Community Dreams Foundation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching CDF records..."):
            result = qa_chain.invoke({"query": prompt})
            response = result["result"]
            sources = result["source_documents"]

            st.markdown(response)
            
            if "Support Form" in response:
                st.link_button("📋 Open Support Form", FORM_LINK, type="primary")

            if sources and "only assist with CDF" not in response:
                with st.expander("📚 View Reference Sources"):
                    for doc in sources:
                        source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        page_num = doc.metadata.get('page', 0) + 1
                        st.write(f"- **Document:** {source_name} (Page {page_num})")

            st.session_state.messages.append({"role": "assistant", "content": response})