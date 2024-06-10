import argparse
import time
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
from request_answer import generate_content

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context with the wisdom, compassion, and insight of a spiritual teacher:

{context}

---

Answer the question based on the above context and provide an answer infused with spiritual guidance: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    start_time = time.time()
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=10)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    #print(prompt)
    gemini_response = generate_content(prompt)
    text = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
    print(text)
    # model = Ollama(model="orca2")
    # response_text = model.invoke(prompt)

    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to execute the function: {elapsed_time} seconds")

    #  return response_text

if __name__ == "__main__":
    main()
