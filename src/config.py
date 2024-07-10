import argparse
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

DEFAULT_MODEL = "OpenAI"
DEFAULT_EMBEDDING_MODEL = "avr/sfr-embedding-mistral:latest"
DEFAULT_PATH = "research"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_NUM_RETRIEVED_DOCS = 10
DEFAULT_OPENAI_API_KEY = "sk-BIWqoUKgrFI6a5ar53E73fA468194104A6644f6d48Af32Da"
DEFAULT_OPENAI_API_BASE = "https://apikeyplus.com/v1"

CONDENSE_QUESTION_TEMPLATE = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""

ANSWER_TEMPLATE = """
### Instruction:
Use the following pieces of context to answer the question at the end. \
If you don't know the answer, just say that you don't know, don't try to make up an answer. \
Keep the answer as concise as possible.

## Research:
{context}

## Question:
{question}
"""

DOCUMENT_TEMPLATE = "{page_content}"

EVALUATION_PROMPT_TEMPLATE = """
### Instruction:
Use the following pieces of context to answer the question at the end in a step-by-step manner showing the calculation steps clearly. The output should follow this format:
[
"operation(",
"value1",
"value2",
")",
"EOF"
]

For example:
subtract("5829", "5735")
EOF

dont't try to add or delete any thing. If you don't know the answer, just say that you don't know, don't try to make up an answer.

## Research:
{context}

## Question:
{question}

## Answer (in step-by-step calculation):
"""
EVALUATION_PROMPT = ChatPromptTemplate.from_template(EVALUATION_PROMPT_TEMPLATE)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run local LLM with RAG with Ollama.")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="The name of the LLM model to use.")
    parser.add_argument("-e", "--embedding_model", default=DEFAULT_EMBEDDING_MODEL, help="The name of the embedding model to use.")
    parser.add_argument("-p", "--path", default=DEFAULT_PATH, help="The path to the directory containing documents to load.")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="The chunk size for text splitting.")
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="The chunk overlap for text splitting.")
    parser.add_argument("--openai_key", default=DEFAULT_OPENAI_API_KEY, help="OpenAI API key for embedding and generation.")
    parser.add_argument("--openai_base", default=DEFAULT_OPENAI_API_BASE, help="OpenAI API base URL for embedding and generation.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation mode to generate prediction file.")
    return parser.parse_args()

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(DOCUMENT_TEMPLATE)
