import os
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root
# This loads your .env file, so OPENAI_API_KEY, CHAT_MODEL, and EMBED_MODEL are accessible.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in .env") # Prevents the script from running without your key.


import os
import pandas as pd
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Load env vars ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in .env")

# --- Load CUAD dataset ---
print("Loading dataset...")
df = pd.read_parquet(
    "hf://datasets/dvgodoy/CUAD_v1_Contract_Understanding_PDF/data/train-00000-of-00001.parquet"
)
# Extract text and create Document objects
# Pulls legal contract text into a dataframe.
# Splits documents into chunks (for embeddings) Using RecursiveCharacterTextSplitter.
# Embeds + builds FAISS vectorstore. Turns chunks into embeddings using OpenAI’s model and stores them in a searchable FAISS index.
# Creates RetrievalQA pipeline. Your LLM (ChatOpenAI) is combined with retriever to answer contract-related questions.

texts = df["text"].dropna().tolist()
docs = [Document(page_content=t, metadata={"source": f"doc_{i}"}) for i, t in enumerate(texts)]

# --- Split documents ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# --- Build embeddings + FAISS vectorstore ---
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --- RAG Pipeline ---
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# --- Quick Test ---
query = "What is the penalty clause in this contract?"
print("Q:", query)
print("A:", qa_chain.run(query))

# Export retriever + chain for app.py
import pickle
with open("vectorstore.pkl", "wb") as f: # This is the important part: it saves your vectorstore locally for later use (like in app.py).
    pickle.dump(vectorstore, f)
