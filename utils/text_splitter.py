from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Split documents into smaller overlapping chunks.

    Args:
        documents: List of LangChain Documents

    Returns:
        List of chunked Documents
    """

    text_splitter = RecursiveCharacterTextSplitter(
        
        chunk_size=1500,
        chunk_overlap=400,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"\nCreated {len(chunks)} text chunks.")

    return chunks