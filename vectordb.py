import os
from langchain_community.vectorstores import Chroma


def create_vector_db(chunks, embeddings, persist_dir="chroma_db"):
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    vectordb.persist()
    print("✅ Vector DB created and saved locally.")

    return vectordb


def load_vector_db(embeddings, persist_dir="chroma_db"):
    if not os.path.exists(persist_dir):
        raise ValueError("❌ No existing DB found. Run ingestion first.")

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    return vectordb
