import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

# Streamlit Page Setup
st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI Knowledge Assistant")
st.markdown("Upload your documents (PDF or TXT) and ask any question related to them.")

# Input API Key
api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# Upload files
uploaded_files = st.file_uploader("📂 Upload Files", type=["pdf", "txt"], accept_multiple_files=True)
build_btn = st.button("⚙️ Build Knowledge Base")

if build_btn and uploaded_files:
    with st.spinner("Processing files..."):
        docs = []
        for file in uploaded_files:
            if file.name.endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file.read())
                    tmp_path = tmp_file.name
                loader = PyPDFLoader(tmp_path)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                    tmp_file.write(file.read())
                    tmp_path = tmp_file.name
                loader = TextLoader(tmp_path)

            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.from_documents(chunks, embeddings)
        st.session_state["db"] = vectordb
        st.success("✅ Knowledge Base Ready!")

# Chat section
if "db" in st.session_state:
    vectordb = st.session_state["db"]
    chat_history = st.session_state.get("chat_history", [])

    query = st.text_input("💬 Ask your question:")
    if st.button("Ask"):
        qa_chain = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), vectordb.as_retriever())
        result = qa_chain({"question": query, "chat_history": chat_history})
        answer = result["answer"]
        chat_history.append((query, answer))
        st.session_state["chat_history"] = chat_history

    if "chat_history" in st.session_state:
        for q, a in st.session_state["chat_history"]:
            st.markdown(f"**🧠 You:** {q}")
            st.markdown(f"**🤖 AI:** {a}")
