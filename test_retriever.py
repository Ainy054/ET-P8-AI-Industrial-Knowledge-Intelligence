from utils.embeddings import get_embedding_model
from utils.vector_store import load_vector_store
from utils.retriever import get_retriever

print("Loading embedding model...")
embedding_model = get_embedding_model()

print("Loading vector store...")
vector_store = load_vector_store(embedding_model)

print("Creating retriever...")
retriever = get_retriever(vector_store)

query = "Why is the pump overheating?"

print(f"\nSearching for: {query}\n")

results = retriever.invoke(query)

print(f"Retrieved {len(results)} documents.\n")

for i, doc in enumerate(results, start=1):
    print("=" * 80)
    print(f"Result {i}")
    print("=" * 80)
    print(doc.page_content[:500])
    print()