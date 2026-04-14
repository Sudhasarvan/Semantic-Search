from ingestion import load_documents, split_documents
from embeding import get_embedding_model
from vectordb import create_vector_db, load_vector_db


def build_db():
    docs = load_documents()
    chunks = split_documents(docs)

    embeddings = get_embedding_model()
    create_vector_db(chunks, embeddings)


def search(query, top_k=3):
    embeddings = get_embedding_model()
    vectordb = load_vector_db(embeddings)

    results = vectordb.similarity_search_with_score(query, k=top_k)

    print("\n🔍 Top Results:\n")

    for i, (doc, score) in enumerate(results):
        print(f"Result {i+1}")
        print(f"Score: {score:.4f}")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(f"Content:\n{doc.page_content[:500]}")
        print("-" * 50)


if __name__ == "__main__":
    while True:
        query_input = input("\nEnter your query (or type 'exit'): ")

        if query_input.lower() == "exit":
            break

        search(query_input)
