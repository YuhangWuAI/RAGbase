import os
import logging
from typing import List
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, JSONLoader
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import parse_arguments, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_OPENAI_API_KEY, DEFAULT_OPENAI_API_BASE
args = parse_arguments()

def load_documents_into_database(model_name: str, documents_path: str, chunk_size: int, chunk_overlap: int) -> Chroma:
    logging.info("Initializing text splitter")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    logging.info("Loading documents")
    raw_documents = load_documents(documents_path)
    documents = text_splitter.split_documents(raw_documents)
    
    logging.info("Creating embeddings and loading documents into Chroma")
    if model_name == "OpenAI":
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=args.openai_key, openai_api_base=args.openai_base)
    else:
        embeddings = OllamaEmbeddings(model=model_name)
    
    db = Chroma.from_documents(documents, embeddings)
    
    logging.info("Documents loaded into Chroma successfully")
    return db

def load_documents(path: str) -> List[Document]:
    if not os.path.exists(path):
        logging.error(f"The specified path does not exist: {path}")
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    loaders = {
        ".pdf": DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True),
        ".md": DirectoryLoader(path, glob="**/*.md", loader_cls=TextLoader, show_progress=True),
        ".json": JSONLoader(file_path="research/demo_test.json", jq_schema='.data[]', text_content=False),
    }

    docs = []
    for file_type, loader in loaders.items():
        logging.info(f"Loading {file_type} files")
        docs.extend(loader.load())
    
    logging.info(f"Total documents loaded: {len(docs)}")
    return docs
