from langchain_core.prompts import PromptTemplate

from providers import OllamaModel
from qdrant.search_in_collection import get_search_result

template = """You are a knowledgeable assistant specializing in human anatomy and physiology.
Your role is to provide accurate, concise, and easy-to-understand explanations about body parts and their functions.
If you does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:
  """

human_template = """
Answer the question according to context:

Context: {context}
Question: {question}
"""

human_prompt = PromptTemplate.from_template(template=human_template)

chain = OllamaModel().create(model="qwen2.5:3b", temperature=0.9, template=template)

while True:
    question = input("Q:")
    search_result = get_search_result(question)
    result = chain.invoke({"input": human_prompt.invoke({"context": search_result, "question": question}).text})
    print(
        "\n",
        result["response"],
        result["history"],
        "\n",
    )
