from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.text_loader import load_documents, split_documents

def get_retriever():
    # Load and split your custom documents
    docs = load_documents()
    split_docs = split_documents(docs)

    # Use HuggingFace sentence transformer
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Use FAISS instead of Chroma to save memory
    vectordb = FAISS.from_documents(split_docs, embedding)

    # Return retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    return retriever
