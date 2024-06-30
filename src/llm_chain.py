from operator import itemgetter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, format_document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string
from langchain.prompts.prompt import PromptTemplate
from config import DEFAULT_NUM_RETRIEVED_DOCS, CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT

def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")

def get_streaming_chain(question: str, memory, llm, db):
    retriever = db.as_retriever(search_kwargs={"k": DEFAULT_NUM_RETRIEVED_DOCS})
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(lambda x: "\n".join(
            [f"{item['role']}: {item['content']}" for item in x["memory"]]
        )),
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        } | CONDENSE_QUESTION_PROMPT | llm
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    answer = final_inputs | ANSWER_PROMPT | llm

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    return final_chain.stream({"question": question, "memory": memory})

def get_chat_chain(llm, db):
    retriever = db.as_retriever(search_kwargs={"k": DEFAULT_NUM_RETRIEVED_DOCS})

    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        } | CONDENSE_QUESTION_PROMPT | llm
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    answer = {
        "answer": final_inputs | ANSWER_PROMPT | llm.with_config(callbacks=[StreamingStdOutCallbackHandler()]),
        "docs": itemgetter("docs"),
    }

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    def chat(question: str):
        inputs = {"question": question}
        result = final_chain.invoke(inputs)
        memory.save_context(inputs, {"answer": result["answer"]})

    return chat
