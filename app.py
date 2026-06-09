# Streamlit runs app.py as a separate process so notebook variables do not
# exist there.
# The entire RAG setup is now defined inside app.py. @st.cache_resource ensures
# the chain is only built once per Streamlit session, not on every interaction.

import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd

# Load API key from environment (set before launching Streamlit)
# In Colab this is set via: os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY environment variable is not set. Set it before launching this app.")
    st.stop()

FAISS_INDEX_PATH = "faiss_legal_index"

@st.cache_resource
def build_chain():
    embeddings = OpenAIEmbeddings()
    if os.path.exists(FAISS_INDEX_PATH):
        db = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        df = pd.read_parquet(
            "hf://datasets/dvgodoy/CUAD_v1_Contract_Understanding_PDF/"
            "data/train-00000-of-00001.parquet"
        )
        texts = df['text'].tolist()
        docs = [Document(page_content=t) for t in texts]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(FAISS_INDEX_PATH)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

st.set_page_config(page_title="Legal Q&A Assistant", page_icon="⚖️")
st.title("⚖️ Legal Q&A Assistant")
st.markdown("Ask any question about the legal contracts in the CUAD dataset.")

query = st.text_input("Your question:", placeholder="e.g. What is the penalty clause?")

if query:
    with st.spinner("Searching documents and generating answer..."):
        qa_chain = build_chain()
        # FIX: .run() is removed in current LangChain. Use .invoke() instead.
        result = qa_chain.invoke({"query": query})
    st.write("### Answer")
    st.write(result["result"])
    with st.expander("Source passages used"):
        for i, doc in enumerate(result["source_documents"], 1):
            st.markdown(f"**Chunk {i}:** {doc.page_content[:300]}...")
