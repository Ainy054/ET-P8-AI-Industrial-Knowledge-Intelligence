from utils.pdf_loader import load_pdf_documents
from utils.text_splitter import split_documents

documents = load_pdf_documents()

chunks = split_documents(documents)

print("\nTotal Chunks:", len(chunks))

print("\nFirst Chunk:\n")

print(chunks[0].page_content)