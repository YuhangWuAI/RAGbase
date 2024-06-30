import sys
from config import parse_arguments
from model_utils import check_model_availability
from document_loader import load_documents_into_database
from llm_chain import get_chat_chain
from langchain_community.llms import Ollama

def main() -> None:
    args = parse_arguments()
    
    try:
        check_model_availability(args.model)
        check_model_availability(args.embedding_model)
    except Exception as e:
        print(e)
        sys.exit()

    try:
        db = load_documents_into_database(args.embedding_model, args.path)
    except FileNotFoundError as e:
        print(e)
        sys.exit()

    llm = Ollama(model=args.model)
    chat = get_chat_chain(llm, db)

    while True:
        try:
            user_input = input("\n\nPlease enter your question (or type 'exit' to end): ")
            if user_input.lower() == "exit":
                break

            chat(user_input)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
