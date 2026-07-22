
import streamlit as st
import re
import os
import time

from langchain_classic.chains import create_retrieval_chain

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from utils.embeddings import get_embedding_model
from utils.vector_store import load_vector_store
from utils.retriever import get_retriever
from utils.llm import get_llm, get_document_chain


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Industrial Knowledge Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
      
def stream_text(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.015)


def parse_answer(answer):

    sections = {
        "Simple Explanation": "",
        "Recommended Actions": "",
        "Source Summary": ""
    }

    current = None

    for line in answer.splitlines():

        line = line.strip()

        if line.startswith("###"):

            title = line.replace("#", "").strip()

            if title in sections:
                current = title
                continue

        if current:
            sections[current] += line + "\n"

    return sections


# --------------------------------------------------
# Load Resources
# --------------------------------------------------

@st.cache_resource
def load_resources():

    embedding_model = get_embedding_model()

    vector_store = load_vector_store(embedding_model)

    retriever = get_retriever(vector_store)

    llm = get_llm()

    document_chain = get_document_chain(llm)

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain
    
def create_uploaded_retriever(uploaded_files):

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    all_documents = []

    for uploaded_file in uploaded_files:

        file_path = upload_dir / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(str(file_path))
        docs = loader.load()

        all_documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=400
    )

    chunks = splitter.split_documents(all_documents)

    embedding_model = get_embedding_model()

    vectorstore = FAISS.from_documents(
        chunks,
        embedding_model
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )



retrieval_chain = load_resources()

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "asked" not in st.session_state:
    st.session_state.asked = False


# --------------------------------------------------
# Header
# --------------------------------------------------


st.title("🏭 Industrial Knowledge Intelligence Platform")

st.divider()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("🏭 Industrial AI")

    st.success("🟢 System Status: Ready")

    st.markdown("---")

    st.subheader("📚 Knowledge Base")

    st.success("6 Industrial Documents Loaded")

    st.markdown("---")

    if not st.session_state.asked:

        st.subheader("💡 Example Questions")

        st.markdown("""
- What causes pump overheating?
- How should mechanical seals be maintained?
- What maintenance is required for reciprocating pumps?
- What inspections are performed monthly?
- What should be done after an industrial incident?
""")

    st.markdown("---")

    st.info("Powered by Gemini + LangChain + FAISS")

    st.markdown("---")

    st.subheader("📂 Upload Knowledge Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(f"✅ {len(uploaded_files)} PDF(s) uploaded")

        st.markdown("**Uploaded Files:**")

        for pdf in uploaded_files:
            st.write(f"📄 {pdf.name}")
    


# --------------------------------------------------
# Question Box
# --------------------------------------------------

st.subheader("💬 Ask Your Question")

question = st.text_area(
    "",
    height=90,
    placeholder="Message Industrial AI..."
)

# --------------------------------------------------
# Ask AI
# --------------------------------------------------

if st.button("🚀 Ask AI", use_container_width=True):

    st.session_state.asked = True

    if not question.strip():
        st.warning("⚠ Please enter a question.")
        st.stop()

    with st.spinner("🔍 Searching documents..."):

        # If user uploaded PDFs, use them
        if uploaded_files:

            uploaded_retriever = create_uploaded_retriever(uploaded_files)

            llm = get_llm()

            document_chain = get_document_chain(llm)

            uploaded_chain = create_retrieval_chain(
                uploaded_retriever,
                document_chain
            )

            response = uploaded_chain.invoke(
                {
                    "input": question
                }
            )

        # Otherwise use default industrial knowledge base
        else:

            response = retrieval_chain.invoke(
                {
                    "input": question
                }
            )

    answer = response["answer"]

    sections = parse_answer(answer)

    # -----------------------------
    # Answer Dashboard
    # -----------------------------

    st.subheader("💡 Simple Explanation")

    text = sections["Simple Explanation"].strip()

    if not text:
        text = answer

    st.write_stream(stream_text(text))

    st.subheader("⚙ Recommended Actions")

    if sections["Recommended Actions"].strip():
        st.markdown(sections["Recommended Actions"])
    else:
        st.write("No specific recommendations available.")

    st.subheader("📄 Source Summary")

    if sections["Source Summary"].strip():
        st.success(sections["Source Summary"])
    else:
        st.write("Information summarized from retrieved documents.")

    st.divider()

    # -----------------------------
    # Retrieved Documents
    # -----------------------------

    with st.expander("📄 Retrieved Documents", expanded=False):

        documents = response.get("context", [])

        if not documents:
            st.info("No retrieved documents available.")

        for i, doc in enumerate(documents, start=1):

            source = os.path.basename(
                doc.metadata.get("source", "Unknown Source")
            )

            st.markdown(f"### 📄 Source {i}")

            st.caption(f"**File:** {source}")

            if "page" in doc.metadata:
                st.caption(f"**Page:** {doc.metadata['page'] + 1}")

            st.text_area(
                "Content",
                doc.page_content,
                height=180,
                disabled=True,
                key=f"doc_{i}"
            )

            st.divider()


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "🏭 Powered by Google Gemini • LangChain • FAISS • HuggingFace Embeddings"
)