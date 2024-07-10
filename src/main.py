import sys
import json
import logging
from config import parse_arguments, DEFAULT_MODEL, DEFAULT_OPENAI_API_KEY, DEFAULT_OPENAI_API_BASE
from model_utils import check_model_availability
from document_loader import load_documents_into_database
from llm_chain import get_chat_chain, get_evaluation_chain
from langchain_community.llms import Ollama
from langchain_community.llms.openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_prediction_steps(response):
    steps = []
    lines = response.splitlines()
    capture = False
    for line in lines:
        if "[" in line:
            capture = True
        if capture:
            line = line.strip().strip('",')
            if line:
                steps.append(line)
        if "]" in line:
            break
    if "EOF" not in steps:
        steps.append("EOF")
    return steps

def main() -> None:
    args = parse_arguments()
    
    if args.model != "OpenAI":
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

    if args.model == "OpenAI":
        llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key=args.openai_key, openai_api_base=args.openai_base)
    else:
        llm = Ollama(model=args.model)

    if args.eval:
        evaluate = get_evaluation_chain(llm, db)
        with open("research/demo_test.json", "r") as f:
            test_data = json.load(f)
        predictions = []
        for item in test_data["data"]:
            question = item["qa"]["question"]
            logging.info(f"Processing question: {question}")
            response = evaluate(question)
            predicted_steps = extract_prediction_steps(response)
            predictions.append({"id": item["id"], "predicted": predicted_steps})
        with open("evaluation/prediction_test.json", "w") as f:
            json.dump(predictions, f, indent=4)
        logging.info("Predictions saved to evaluation/prediction_test.json")
    else:
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
