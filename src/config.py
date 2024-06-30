import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run local LLM with RAG with Ollama.")
    parser.add_argument("-m", "--model", default="mistral", help="The name of the LLM model to use.")
    parser.add_argument("-e", "--embedding_model", default="llama3:8b", help="The name of the embedding model to use.")
    parser.add_argument("-p", "--path", default="research", help="The path to the directory containing documents to load.")
    return parser.parse_args()
