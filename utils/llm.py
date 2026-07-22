from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from utils.config import GOOGLE_API_KEY


def get_llm():

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )

    print("✅ Gemini loaded successfully.")

    return llm


def get_document_chain(llm):

    prompt = ChatPromptTemplate.from_template("""
You are an Industrial AI Knowledge Assistant.

You are answering questions using retrieved industrial manuals.

IMPORTANT:

- The retrieved text may contain OCR errors.
- Information may be spread across multiple document chunks.
- Combine related information from all retrieved chunks.
- Do NOT simply say "the context does not specify" if related evidence exists.
- Answer naturally in professional English.
- Never invent facts that are completely unsupported.
- If the answer is not explicitly stated but can be reasonably inferred from multiple retrieved passages, clearly mention that it is an inference based on the retrieved documents.
- Prefer practical maintenance recommendations over simply stating that information is missing.

<context>
{context}
</context>

Question:
{input}

Return the answer in this format:

### Simple Explanation
Give a clear explanation.

### Recommended Actions
Provide practical recommendations.

### Risk Level
Low / Medium / High

### Confidence
Give a percentage.

### Source Summary
Mention which manuals or reports the information came from.
""")
    

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    return document_chain