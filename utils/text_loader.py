from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_documents():
    folder = "knowledge_base"
    docs = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(folder, filename))
            docs.extend(loader.load())
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(docs)
