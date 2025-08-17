import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# --- Load env ---
load_dotenv()
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# --- Load vectorstore ---
with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# --- UI ---
st.set_page_config(page_title="Legal Q&A Assistant", page_icon="⚖️")
st.title("⚖️ Legal Q&A Assistant")

query = st.text_input("Ask a question about the legal document:")
if st.button("Ask") and query:
    try:
        answer = qa_chain.run(query)
        st.markdown("### Answer")
        st.write(answer)
    except Exception as e:
        st.error(f"Error: {e}")
# Save the vectorstore for use in app.py