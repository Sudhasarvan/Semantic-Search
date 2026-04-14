import streamlit as st
from ingestion import load_documents, split_documents
from embeding import get_embedding_model
from vectordb import create_vector_db, load_vector_db
import os

st.set_page_config(page_title="Local Semantic Search", layout="wide")

st.title("🔍 Local Document Semantic Search")
st.write("Search your documents using HuggingFace embeddings + ChromaDB (Fully Offline)")

PERSIST_DIR = "chroma_db"


# -----------------------------
# Build Database
# -----------------------------
import streamlit as st

if st.button("📥 Build / Rebuild Database"):
    docs = load_documents()
    chunks = split_documents(docs)

    embeddings = get_embedding_model()
    create_vector_db(chunks, embeddings)

    st.success("✅ Database built successfully!")

# -----------------------------
# Search Section
# -----------------------------
st.subheader("🔎 Search Your Documents")

query = st.text_input("Enter your query:")

top_k = st.slider("Number of results", 1, 10, 3)


if st.button("Search"):
    if not os.path.exists(PERSIST_DIR):
        st.error("❌ Please build the database first!")
    elif query.strip() == "":
        st.warning("⚠️ Enter a query")
    else:
        with st.spinner("Searching..."):
            embeddings = get_embedding_model()
            vectordb = load_vector_db(embeddings, PERSIST_DIR)

            results = vectordb.similarity_search_with_score(query, k=top_k)

        st.success(f"✅ Found {len(results)} results")

        for i, (doc, score) in enumerate(results):
            with st.expander(f"Result {i+1} | Score: {score:.4f}"):
                st.write(f"📄 Source: {doc.metadata.get('source', 'N/A')}")
                st.write(doc.page_content)
