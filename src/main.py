import sys
import logging
from config import parse_arguments, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from model_utils import check_model_availability
from document_loader import load_documents_into_database
from llm_chain import get_chat_chain
from langchain_community.llms import Ollama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main() -> None:
    args = parse_arguments()
    
    try:
        logging.info(f"Checking availability of model: {args.model}")
        check_model_availability(args.model)
        logging.info(f"Checking availability of embedding model: {args.embedding_model}")
        check_model_availability(args.embedding_model)
    except Exception as e:
        logging.error(e)
        sys.exit()

    try:
        logging.info(f"Loading documents from path: {args.path}")
        db = load_documents_into_database(args.embedding_model, args.path, args.chunk_size, args.chunk_overlap)
        logging.info("Documents loaded and database created successfully")
    except FileNotFoundError as e:
        logging.error(e)
        sys.exit()

    llm = Ollama(model=args.model)
    chat = get_chat_chain(llm, db)

    while True:
        try:
            user_input = input("\n\nPlease enter your question (or type 'exit' to end): ")
            if user_input.lower() == "exit":
                break

            response = chat(user_input)
            logging.info(f"User question: {user_input}")
            logging.info(f"Assistant response: {response}")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
