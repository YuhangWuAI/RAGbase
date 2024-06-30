from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, JSONLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
from typing import List


def load_documents_into_database(model_name: str, documents_path: str, chunk_size: int, chunk_overlap: int) -> Chroma:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print("Loading documents")
    raw_documents = load_documents(documents_path)
    documents = text_splitter.split_documents(raw_documents)

    print("Creating embeddings and loading documents into Chroma")
    db = Chroma.from_documents(documents, OllamaEmbeddings(model=model_name))
    return db

def load_documents(path: str) -> List[Document]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    loaders = {
        ".pdf": DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True),
        ".md": DirectoryLoader(path, glob="**/*.md", loader_cls=TextLoader, show_progress=True),
        ".json": JSONLoader(file_path="research/test.json", jq_schema='.data[]', text_content=False),
    }

    docs = []
    for file_type, loader in loaders.items():
        print(f"Loading {file_type} files")
        docs.extend(loader.load())
    return docs
