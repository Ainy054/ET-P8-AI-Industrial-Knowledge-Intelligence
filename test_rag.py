from utils.embeddings import get_embedding_model
from utils.vector_store import load_vector_store
from utils.retriever import get_retriever
from utils.llm import get_llm, create_rag_chain

print("Loading embedding model...")
embedding_model = get_embedding_model()

print("Loading vector store...")
vector_store = load_vector_store(embedding_model)

print("Creating retriever...")
retriever = get_retriever(vector_store)

print("Loading Gemini...")
llm = get_llm()

chain = create_rag_chain(llm)

question = input("\nAsk your question: ")

docs = retriever.invoke(question)

context = "\n\n".join([doc.page_content for doc in docs])

print("\n" + "=" * 80)
print("RETRIEVED CONTEXT")
print("=" * 80)
print(context[:3000])   # Print first 3000 characters
print("=" * 80 + "\n")

response = chain.invoke({
    "context": context,
    "question": question
})

print("\n")
print("=" * 80)
print("AI RESPONSE")
print("=" * 80)
print(response.content)