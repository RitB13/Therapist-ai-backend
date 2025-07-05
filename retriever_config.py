from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from utils.text_loader import load_documents, split_documents

def get_retriever():
    docs = load_documents()
    split_docs = split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(split_docs, embedding)
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    return retriever
