import json
import logging
from config import parse_arguments, DEFAULT_MODEL, DEFAULT_OPENAI_API_KEY, DEFAULT_OPENAI_API_BASE
from model_utils import check_model_availability
from document_loader import load_documents_into_database
from langchain_community.llms import Ollama
from langchain_community.llms.openai import OpenAI
from llm_chain import get_evaluation_chain

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_predictions(test_data, evaluate):
    predictions = []
    for item in test_data["data"]:
        question = item["qa"]["question"]
        logging.info(f"Processing question: {question}")
        response = evaluate(question)
        predicted_steps = extract_prediction_steps(response)
        predictions.append({"id": item["id"], "predicted": predicted_steps})
    return predictions

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

def main():
    args = parse_arguments()
    if not args.eval:
        logging.error("This script should be run with --eval flag.")
        return

    # Load the test dataset
    with open("research/demo_test.json", "r") as f:
        test_data = json.load(f)

    if args.model != "OpenAI":
        # Check model availability
        logging.info(f"Checking availability of model: {args.model}")
        check_model_availability(args.model)
        logging.info(f"Checking availability of embedding model: {args.embedding_model}")
        check_model_availability(args.embedding_model)

    # Load documents into database
    logging.info(f"Loading documents from path: {args.path}")
    db = load_documents_into_database(args.embedding_model, args.path, args.chunk_size, args.chunk_overlap)
    logging.info("Documents loaded and database created successfully")

    if args.model == "OpenAI":
        llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key=args.openai_key, openai_api_base=args.openai_base)
    else:
        llm = Ollama(model=args.model)

    evaluate = get_evaluation_chain(llm, db)

    # Generate predictions
    predictions = generate_predictions(test_data, evaluate)

    # Save predictions to a file
    with open("evaluation/prediction_test.json", "w") as f:
        json.dump(predictions, f, indent=4)
    logging.info("Predictions saved to evaluation/prediction_test.json")

if __name__ == "__main__":
    main()
