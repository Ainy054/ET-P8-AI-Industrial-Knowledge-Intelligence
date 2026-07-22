from utils.pdf_loader import load_pdf_documents
from utils.text_splitter import split_documents
from utils.embeddings import get_embedding_model
from utils.vector_store import create_vector_store

print("Loading PDFs...")
documents = load_pdf_documents()

print("Splitting...")
chunks = split_documents(documents)

print("Loading Embedding Model...")
embedding_model = get_embedding_model()

print("Creating Vector Store...")
vector_store = create_vector_store(
    chunks,
    embedding_model
)

print("\n🎉 Everything completed successfully!")