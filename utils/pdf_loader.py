import os
from langchain_community.document_loaders import PyPDFLoader
from utils.config import DOCUMENTS_PATH


def load_pdf_documents():
    """
    Load all PDF files from the documents folder.

    Returns:
        list: LangChain Document objects
    """

    documents = []

    if not os.path.exists(DOCUMENTS_PATH):
        raise FileNotFoundError(f"Folder '{DOCUMENTS_PATH}' not found.")

    pdf_files = [f for f in os.listdir(DOCUMENTS_PATH) if f.endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in documents folder.")

    print(f"\nFound {len(pdf_files)} PDF files.\n")

    for pdf in pdf_files:
        pdf_path = os.path.join(DOCUMENTS_PATH, pdf)

        print(f"Loading: {pdf}")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        documents.extend(pages)

        print(f"Loaded {len(pages)} pages")

    print(f"\nTotal Pages Loaded: {len(documents)}")

    return documents